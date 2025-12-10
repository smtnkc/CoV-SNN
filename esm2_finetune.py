#!/usr/bin/env python3

"""
python esm2_finetune.py \
  --csv data/unique_Omicron_2k.csv \
  --seq-col sequence \
  --take-top 2000 \
  --model esm2_t6_8M_UR50D \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-5 \
  --out-ckpt checkpoints/esm2_t6_8M_UR50D_omicron.pt \
  --mask-mode every_pos \
  --amp
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

def read_sequences_from_csv(path: str, seq_col: str, take_top: int):
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        for c in ("seq","sequence","protein","aa","aa_seq"):
            if c in df.columns:
                seq_col = c
                break
        else:
            seq_col = df.columns[0]
    take_n = min(int(take_top), len(df))
    seqs = df.iloc[:take_n][seq_col].astype(str).tolist()
    return [s.strip().upper().replace(" ","") for s in seqs if isinstance(s,str) and s.strip()]

def batched(iterable, n):
    b = []
    for x in iterable:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b

def load_esm(model_name: str, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = torch.hub.load('facebookresearch/esm:main', model_name)
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, device

def make_mlm_batch_random(toks, alphabet, mask_prob=0.15):
    """
    toks: (B, L) int64
    Returns:
      masked_toks: (B, L)
      labels: (B, L) with -100 where we do NOT compute loss
    """
    device = toks.device
    pad_idx = alphabet.padding_idx
    mask_idx = alphabet.mask_idx

    masked = toks.clone()
    labels = toks.clone()

    B, L = toks.shape
    labels[:] = -100  # default ignore_index

    rng = torch.Generator(device=device)
    # rng.manual_seed(12345) # Don't fix seed here to allow randomness across epochs

    for i in range(B):
        row = toks[i]
        nonpad = (row != pad_idx).sum().item()
        # Exclude special tokens at beginning/end, same logic as PLL
        L_eff = max(nonpad - 1, 2)
        interior = torch.arange(1, L_eff, device=device)
        if len(interior) == 0:
            continue
        n_mask = max(1, int(len(interior) * mask_prob))
        perm = torch.randperm(len(interior), generator=rng, device=device)
        chosen = interior[perm[:n_mask]]

        labels[i, chosen] = row[chosen]      # we supervise only on these
        masked[i, chosen] = mask_idx         # replace input with [MASK]

    return masked, labels

def generate_every_pos_samples(sequences):
    """
    Generator that yields (label, sequence_with_one_mask_info) tuples.
    But wait, batch_converter expects raw sequences.
    
    Strategy:
    For each sequence S of length L:
      Yield L copies of S, but we need to know WHICH position to mask.
      
    Alternative:
    Yield (sequence, mask_pos_index)
    """
    for i, seq in enumerate(sequences):
        # seq is string
        # We want to mask every position from 0 to len(seq)-1
        # But wait, ESM adds CLS at 0 and EOS at end.
        # So if seq is "ABC", tokens are [CLS, A, B, C, EOS]
        # We want to mask A (idx 1), B (idx 2), C (idx 3).
        
        L = len(seq)
        for pos in range(L):
            # Yield the sequence and the 0-based index in the original string to mask
            # The model function will map this to token index (pos + 1)
            yield (f'seq{i}_pos{pos}', seq, pos)

def make_mlm_batch_deterministic(chunk_data, alphabet, batch_converter, device):
    """
    chunk_data: list of (label, seq_str, mask_pos_int)
    """
    # Unpack
    raw_labels = [x[0] for x in chunk_data]
    seqs = [x[1] for x in chunk_data]
    mask_pos_list = [x[2] for x in chunk_data]
    
    # Tokenize
    # batch_converter takes list of (label, seq)
    # We can just pass dummy labels
    batch_input = list(zip(raw_labels, seqs))
    _, _, toks = batch_converter(batch_input)
    toks = toks.to(device)
    
    mask_idx = alphabet.mask_idx
    
    masked = toks.clone()
    labels = toks.clone()
    labels[:] = -100
    
    B = toks.shape[0]
    
    for i in range(B):
        # mask_pos is 0-based index in the sequence string.
        # In token tensor, CLS is at 0. So token index is mask_pos + 1.
        # Verify: "ABC" -> [CLS, A, B, C, EOS]. A is at 1. mask_pos=0 -> 1. Correct.
        p = mask_pos_list[i] + 1
        
        # Safety check
        if p < toks.shape[1] - 1: # Ensure not overwriting EOS or out of bounds
            labels[i, p] = toks[i, p]
            masked[i, p] = mask_idx
            
    return masked, labels

def main():
    ap = argparse.ArgumentParser(description="Fine-tune ESM-2 on Omicron sequences (MLM style)")
    ap.add_argument('--csv', required=True, help='CSV with Omicron sequences')
    ap.add_argument('--seq-col', default='sequence')
    ap.add_argument('--take-top', type=int, default=20000)
    ap.add_argument('--model', default='esm2_t33_650M_UR50D')
    ap.add_argument('--out-ckpt', required=True, help='Path to save fine-tuned state_dict')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--mask-prob', type=float, default=0.15)
    ap.add_argument('--mask-mode', choices=['random', 'every_pos'], default='random', help='random: standard MLM. every_pos: systematic 1-mask per pos (CovSNN style)')
    ap.add_argument('--mask-stride', type=int, default=1, help='For every_pos mode: mask every Nth position (default 1 = all positions). Set to 10 to reduce data 10x.')
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()

    seqs = read_sequences_from_csv(args.csv, args.seq_col, args.take_top)
    if not seqs:
        raise ValueError("No sequences loaded from Omicron CSV.")

    model, alphabet, batch_converter, device = load_esm(args.model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_steps = 0
        
        if args.mask_mode == 'random':
            # Standard Random MLM
            labeled = [(f'seq{i}', s) for i, s in enumerate(seqs)]
            np.random.shuffle(labeled)
            
            iterator = tqdm(batched(labeled, args.batch_size), desc=f'Epoch {epoch+1}/{args.epochs} (Random MLM)')
            for chunk in iterator:
                _, _, toks = batch_converter(chunk)
                toks = toks.to(device)
                masked_toks, labels = make_mlm_batch_random(toks, alphabet, mask_prob=args.mask_prob)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = model(masked_toks, repr_layers=[], return_contacts=False)
                    logits = out["logits"]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_steps += 1
                
        else:
            # Systematic Every-Pos MLM
            # We need to shuffle the (seq, pos) pairs to avoid batching same sequence together too much?
            # Or maybe it's better to batch same sequence together for padding efficiency?
            # CovSNN likely shuffled. Let's generate all then shuffle?
            # 2000 seqs * ~1200 len = 2.4M samples. 
            # Generating list might take memory. 
            # Let's generate list of (seq_idx, pos) indices, shuffle that, then fetch seqs.
            
            print(f"Generating mask indices (stride={args.mask_stride})...")
            all_tasks = []
            for i, s in enumerate(seqs):
                L = len(s)
                # Stride logic: mask 0, 10, 20...
                for p in range(0, L, args.mask_stride):
                    all_tasks.append((i, p))
            
            print(f"Total training samples: {len(all_tasks)}")
            random.shuffle(all_tasks)
            
            # Create a generator that yields chunks of tasks
            def task_generator():
                batch = []
                for seq_idx, pos in all_tasks:
                    batch.append((f'seq{seq_idx}_p{pos}', seqs[seq_idx], pos))
                    if len(batch) == args.batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            iterator = tqdm(task_generator(), total=len(all_tasks)//args.batch_size, desc=f'Epoch {epoch+1}/{args.epochs} (Strided MLM)')
            
            for chunk in iterator:
                # chunk is list of (label, seq, pos)
                masked_toks, labels = make_mlm_batch_deterministic(chunk, alphabet, batch_converter, device)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = model(masked_toks, repr_layers=[], return_contacts=False)
                    logits = out["logits"]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch+1}: avg MLM loss = {avg_loss:.4f}")

    os.makedirs(os.path.dirname(args.out_ckpt) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"Saved fine-tuned weights to {args.out_ckpt}")

if __name__ == '__main__':
    main()
