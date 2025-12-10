#!/usr/bin/env python3
"""
Alignment-free CSCS with ESM-2.

Example
-------
python esm2_compute_cscs.py \                                                                                                                                                (pytorch38) 
    --csv data/unique_Omicron_2k.csv:omicron \
    --csv data/unique_Eris_2k.csv:eris \
    --csv data/unique_New_2k.csv:new \
    --csv data/unique_Gpt_1.0_2k.csv:gpt \
    --seq-col sequence \
    --take-top 2000 \
    --eris-label eris \
    --model esm2_t12_35M_UR50D \
    --model-ckpt checkpoints/esm2_t12_35M_UR50D_omicron.pt \
    --pll-model esm2_t12_35M_UR50D \
    --pll-model-ckpt checkpoints/esm2_t12_35M_UR50D_omicron.pt \
    --pll-mode stride \
    --pll-stride 10 \
    --pll-max-bsz 64 \
    --ref-center mean:variant \
    --ref-variant omicron \
    --target-variant new \
    --target-variant eris \
    --target-variant gpt \
    --semantic-change-mode l2_pos \
    --embed-batch-size 4 \
    --cscs-formula rank_sum \
    --amp \
    --compile \
    --out results/esm2_35M_epoch1_l2pos_cscs.csv
"""
import argparse
import os
import math
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

def read_sequences_from_csv(path: str, seq_col: str, take_top: int):
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        for c in ("seq","sequence","protein","aa","aa_seq"):
            if c in df.columns: seq_col=c; break
        else: seq_col = df.columns[0]
    take_n = min(int(take_top), len(df))
    seqs = df.iloc[:take_n][seq_col].astype(str).tolist()
    return [s.strip().upper().replace(" ","") for s in seqs if isinstance(s,str) and s.strip()]

def load_seq_from_file(path: str) -> str:
    with open(path,'r') as f:
        for line in f:
            s=line.strip().upper()
            if s and not s.startswith('>'): return s
    raise ValueError(f'No usable sequence lines in {path}')

def batched(iterable, n):
    b=[]
    for x in iterable:
        b.append(x)
        if len(b)==n: yield b; b=[]
    if b: yield b

def load_esm(model_name: str, device: str=None, ckpt_path: str=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = torch.hub.load('facebookresearch/esm:main', model_name)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, device


@torch.no_grad()
def embed_mean_token(model, batch_converter, device, sequences, repr_layer=None, amp=False, batch_size=16):
    if repr_layer is None: repr_layer = getattr(model, 'num_layers', 33)
    reps=[]
    labeled=[(f'seq{i}', s) for i,s in enumerate(sequences)]
    for chunk in tqdm(batched(labeled, batch_size), desc='Embedding (mean token repr)'):
        _,_,toks = batch_converter(chunk); toks=toks.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(toks, repr_layers=[repr_layer], need_head_weights=False)
            token_reps = out['representations'][repr_layer]
            for tr in token_reps: reps.append(tr[1:-1].mean(dim=0).float().cpu().numpy())
    return np.vstack(reps)

@torch.no_grad()
def pll_cross_packed(model, alphabet, batch_converter, device, sequences, mode='all', stride=3, random_k=120, first_n=120, max_bsz=256, amp=False):
    N=len(sequences)
    labeled=[(f'seq{i}', s) for i,s in enumerate(sequences)]
    _,_,toks_all = batch_converter(labeled); toks_all=toks_all.to(device)
    pad_idx = alphabet.padding_idx
    rng = np.random.RandomState(12345)

    pos_lists=[]; total_positions=0
    for i in range(N):
        toks=toks_all[i]; nonpad=(toks!=pad_idx).sum().item()
        L_eff=max(nonpad-1,2); interior=list(range(1,L_eff))
        if   mode=='stride':  pos_list=interior[::max(1,stride)]
        elif mode=='firstN':  pos_list=interior[:first_n]
        elif mode=='random':  pos_list=interior if len(interior)<=random_k else sorted(rng.choice(interior, size=random_k, replace=False).tolist())
        else:                 pos_list=interior
        pos_lists.append(pos_list); total_positions += len(pos_list)

    queue=[]
    for i,plist in enumerate(pos_lists):
        for p in plist: queue.append((i,p))

    sums=np.zeros(N,dtype=np.float64); counts=np.zeros(N,dtype=np.int64)
    for start in tqdm(range(0,len(queue), max_bsz), desc=f'Grammaticality (PLL, packed:{mode})'):
        chunk=queue[start:start+max_bsz]; B=len(chunk)
        toks_masked = toks_all.new_zeros((B, toks_all.shape[1]))
        true_tok_idx=[]; pos_tensor_list=[]
        for row,(i,p) in enumerate(chunk):
            base=toks_all[i].clone()
            true_tok_idx.append(base[p].item())
            base[p]=alphabet.mask_idx
            toks_masked[row]=base
            pos_tensor_list.append(p)
        true_tok_idx=torch.tensor(true_tok_idx, device=device)
        pos_tensor=torch.tensor(pos_tensor_list, device=device)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(toks_masked, repr_layers=[], return_contacts=False)
            logits=out['logits']
            sel = logits[torch.arange(B, device=device), pos_tensor, :]
            logp = F.log_softmax(sel, dim=-1).gather(1, true_tok_idx.view(-1,1)).squeeze(1).float()
        for (i,_), lp in zip(chunk, logp.tolist()): sums[i]+=lp; counts[i]+=1
    out=np.full(N, np.nan, dtype=np.float32); m=counts>0
    out[m]=(sums[m]/counts[m]).astype(np.float32); return out

def percentile_rank(x, higher_is_better=True):
    order=np.argsort(x); ranks=np.empty_like(order, dtype=np.float64); ranks[order]=np.arange(len(x), dtype=np.float64)
    pr=ranks/max(len(x)-1,1); return 1.0-pr if higher_is_better else pr

def cscs_from_formula(semantic_change, pll, formula):
    """
    semantic_change : numpy array >= 0
    pll             : numpy array <= 0  (average log-prob in natural log base)
    """
    if formula == 'log_ratio':
        # We want: log10(P(grammaticality)) - log10(semantic_change)
        # pll is natural-log prob (<=0). Convert to base-10 without exp for stability.
        eps = 1e-12
        log10_prob = pll / np.log(10.0)  # convert ln(.) to log10(.)
        return log10_prob - np.log10(np.maximum(semantic_change, eps))

    elif formula == 'percentile_hmean':
        sem_pr = percentile_rank(semantic_change, True)
        gram_pr = percentile_rank(pll, True)
        eps = 1e-9
        return 2.0 / (np.maximum(sem_pr, eps) ** -1 + np.maximum(gram_pr, eps) ** -1)

    elif formula == 'rank_sum':
        # Rank PLL descending (high prob is good) -> Rank 0 is best
        # Rank SC descending (high change is good) -> Rank 0 is best
        # We want to minimize rank sum, or maximize -(rank sum)
        
        # argsort gives indices that sort the array. argsort(argsort) gives the rank (0..N-1)
        # For PLL, higher is better.
        # sort descending: -pll
        rank_pll = np.argsort(np.argsort(-pll))
        
        # For Semantic Change, higher is better (novelty).
        # sort descending: -semantic_change
        rank_sc = np.argsort(np.argsort(-semantic_change))
        
        # Sum of ranks (lower is better)
        r_sum = rank_pll + rank_sc
        
        # Return negative rank sum so that higher CSCS score is better
        return -r_sum.astype(np.float32)

    else:
        raise ValueError(f'Unknown --cscs-formula {formula}')


def main():
    ap=argparse.ArgumentParser(description='Alignment-free CSCS with ESM-2 (exact PLL supported)')
    ap.add_argument('--csv', action='append', required=True, help='CSV path with optional label: file.csv[:label]. Repeatable.')
    ap.add_argument('--seq-col', default='sequence'); ap.add_argument('--take-top', type=int, default=2000)
    ap.add_argument('--eris-label', required=True)
    ap.add_argument('--model', default='esm2_t33_650M_UR50D'); ap.add_argument('--pll-model', default=None)
    ap.add_argument('--model-ckpt', default=None, help='Optional path to fine-tuned state_dict for --model')
    ap.add_argument('--pll-model-ckpt', default=None, help='Optional path to fine-tuned state_dict for --pll-model')
    ap.add_argument('--repr-layer', type=int, default=None); ap.add_argument('--amp', action='store_true')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--pll-mode', choices=['all','random','stride','firstN'], default='all')
    ap.add_argument('--pll-stride', type=int, default=3); ap.add_argument('--pll-random-k', type=int, default=120)
    ap.add_argument('--pll-first-n', type=int, default=120); ap.add_argument('--pll-max-bsz', type=int, default=256)
    ap.add_argument('--embed-batch-size', type=int, default=4, help='Batch size for embedding (lower this if OOM)')
    ap.add_argument('--ref-center', default='mean:variant', help='mean:variant|seq:<AASEQ>|file:<PATH>|wt. If mean:variant, uses --ref-variant to find sequences.')
    ap.add_argument('--ref-variant', help='Label of the variant to use as reference center (if ref-center is mean:variant)')
    ap.add_argument('--target-variant', action='append', required=True, help='Label(s) of variants to rank. PLL/CSCS will ONLY be calculated for these.')
    ap.add_argument('--semantic-change-mode', choices=['l1_mean', 'l2_mean', 'l2_pos'], default='l2_mean', help='Distance metric for semantic change.')
    ap.add_argument('--pad-to', type=int, default=None, help='Fixed length for l2_pos mode. If None, uses max length of data.')
    ap.add_argument('--wt-seq', default=None); ap.add_argument('--wt-file', default=None)
    ap.add_argument('--cscs-formula', choices=['log_ratio','percentile_hmean', 'rank_sum'], default='log_ratio')
    # ap.add_argument('--exclude-label', action='append', help='Exclude sequences with this label from the final ranking (e.g. the reference group). Repeatable.')
    ap.add_argument('--out', required=True)
    args=ap.parse_args()

    # Parse CSVs
    sources=[]
    for spec in args.csv:
        if ':' in spec: path,label=spec.split(':',1)
        else: path,label=spec, os.path.splitext(os.path.basename(spec))[0]
        if not os.path.exists(path): raise FileNotFoundError(path)
        sources.append((path,label))

    # Load sequences
    seqs=[]; labels=[]; per_source_counts={}
    for path,label in sources:
        cur=read_sequences_from_csv(path, args.seq_col, args.take_top)
        seqs.extend(cur); labels.extend([label]*len(cur)); per_source_counts[label]=len(cur)
    if not seqs: raise ValueError('No sequences loaded.')

    # Load models
    model, alphabet, batch_converter, device = load_esm(
        args.model,
        ckpt_path=args.model_ckpt
    )
    if args.repr_layer is None:
        args.repr_layer = getattr(model, 'num_layers', 33)
        print(f"Using default repr_layer: {args.repr_layer}")

    pll_model_name = args.pll_model if args.pll_model else args.model
    pll_model, pll_alpha, pll_batch_conv, _ = load_esm(
        pll_model_name,
        device=device,
        ckpt_path=args.pll_model_ckpt
    )

    if args.compile:
        try: pll_model = torch.compile(pll_model, mode='reduce-overhead')
        except Exception: pass

    # ---------------------------------------------------------
    # Semantic Change Calculation
    # ---------------------------------------------------------
    
    print(f"Computing Semantic Change ({args.semantic_change_mode})...")
    
    # Helper to embed sequences based on mode
    def get_embeddings(sequence_list, mode):
        if mode in ['l1_mean', 'l2_mean']:
            return embed_mean_token(
                model, batch_converter, device, sequence_list,
                repr_layer=args.repr_layer, amp=args.amp, batch_size=args.embed_batch_size
            )
        elif mode == 'l2_pos':
            # For l2_pos, we need full tensors padded to fixed length
            # Determine max length
            max_len = args.pad_to if args.pad_to else max(len(s) for s in seqs)
            print(f"Embedding full tensors (padded to {max_len})...")
            
            reps = []
            
            labeled = [(f'seq{i}', s[:max_len]) for i, s in enumerate(sequence_list)] # Truncate if needed
            
            # Batch processing
            batch_size = args.embed_batch_size
            res = []
            
            for chunk in tqdm(batched(labeled, batch_size), desc='Embedding (full tensor)'):
                labels, strs, toks = batch_converter(chunk)
                # Pad toks to max_len + 2 (cls/eos) manually if batch_converter didn't reach it?
                # batch_converter pads to longest in batch. We need global alignment.
                # So we must pad the *output representations* or ensure input is padded.
                # Easier to pad output representations to max_len.
                
                toks = toks.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = model(toks, repr_layers=[args.repr_layer], need_head_weights=False)
                    token_reps = out['representations'][args.repr_layer] # (B, L_batch, D)
                
                for i, tr in enumerate(token_reps):
                    # tr is (L_seq, D). Remove CLS/EOS -> (L_seq-2, D)
                    # But wait, we want to preserve alignment. 
                    # If we remove CLS/EOS, we have the sequence.
                    seq_rep = tr[1:-1] # (L_actual, D)
                    
                    # Pad to max_len
                    curr_len = seq_rep.shape[0]
                    if curr_len < max_len:
                        pad_amt = max_len - curr_len
                        # Zero pad
                        pad = torch.zeros((pad_amt, seq_rep.shape[1]), device=seq_rep.device, dtype=seq_rep.dtype)
                        seq_rep = torch.cat([seq_rep, pad], dim=0)
                    elif curr_len > max_len:
                        seq_rep = seq_rep[:max_len]
                        
                    res.append(seq_rep.float().detach().cpu().numpy())
            
            return np.stack(res) # (N, max_len, D)
            
    # Helper that embeds a single sequence (for reference)
    def _embed_ref_seq(ref_seq: str):
        return get_embeddings([ref_seq], args.semantic_change_mode)[0]

    ref_spec = args.ref_center.strip()
    
    # 1. Calculate Reference Vector
    if ref_spec == 'mean:variant':
        if not args.ref_variant:
            raise ValueError("If --ref-center is mean:variant, you must specify --ref-variant <LABEL>")
        
        # Find sequences matching ref_variant
        ref_idxs = [i for i, l in enumerate(labels) if l == args.ref_variant]
        if not ref_idxs: raise ValueError(f'No sequences found for --ref-variant {args.ref_variant}')
        
        print(f"Computing reference center from {len(ref_idxs)} sequences (label={args.ref_variant})...")
        ref_seqs_list = [seqs[i] for i in ref_idxs]
        
        ref_reps = get_embeddings(ref_seqs_list, args.semantic_change_mode)
        ref_vec = ref_reps.mean(axis=0) # Mean of (N, D) or (N, L, D) -> (D,) or (L, D)
            
    elif ref_spec.startswith('seq:'):
        ref_seq = ref_spec.split(':', 1)[1].strip().upper().replace(' ', '')
        ref_vec = _embed_ref_seq(ref_seq)
        
    elif ref_spec.startswith('file:'):
        ref_path = ref_spec.split(':', 1)[1].strip()
        ref_seq = load_seq_from_file(ref_path)
        ref_vec = _embed_ref_seq(ref_seq)
        
    elif ref_spec == 'wt':
        if args.wt_seq: ref_s = args.wt_seq
        elif args.wt_file: ref_s = load_seq_from_file(args.wt_file)
        else: raise ValueError("Need wt")
        ref_vec = _embed_ref_seq(ref_s)
    else:
        raise ValueError(f"Unknown ref spec {ref_spec}")

    # 2. Filter for Targets
    target_idxs = [i for i, l in enumerate(labels) if l in args.target_variant]
    if not target_idxs:
        raise ValueError(f"No sequences found for --target-variant {args.target_variant}")
    
    print(f"Filtering for targets: {args.target_variant}. Keeping {len(target_idxs)} sequences.")
    
    target_seqs = [seqs[i] for i in target_idxs]
    target_labels = [labels[i] for i in target_idxs]
    
    # 3. Compute Embeddings for Targets ONLY
    print(f"Embedding {len(target_seqs)} target sequences...")
    target_reps = get_embeddings(target_seqs, args.semantic_change_mode)
    
    # 4. Compute Distance
    diff = target_reps - ref_vec # (N, D) or (N, L, D)
    
    if args.semantic_change_mode == 'l2_mean':
        semantic_change = np.linalg.norm(diff, axis=1) # L2 norm
    elif args.semantic_change_mode == 'l1_mean':
        semantic_change = np.sum(np.abs(diff), axis=1) # L1 norm
    elif args.semantic_change_mode == 'l2_pos':
        # diff is (N, L, D). Flatten to (N, L*D) then norm
        diff_flat = diff.reshape(diff.shape[0], -1)
        semantic_change = np.linalg.norm(diff_flat, axis=1)

    # 5. Grammaticality via packed PLL (Targets ONLY)
    pll = pll_cross_packed(
        pll_model, pll_alpha, pll_batch_conv, device, target_seqs,
        mode=args.pll_mode, stride=args.pll_stride, random_k=args.pll_random_k,
        first_n=args.pll_first_n, max_bsz=args.pll_max_bsz, amp=args.amp
    )

    # CSCS
    cscs=cscs_from_formula(semantic_change, pll, args.cscs_formula)

    # Ranking & stats
    df=pd.DataFrame({'label':target_labels,'sequence':target_seqs,'semantic_change':semantic_change,'pll_logprob':pll,'cscs':cscs}).sort_values('cscs', ascending=False, kind='mergesort')
    df['rank']=np.arange(1,len(df)+1); df['percentile_global']=(len(df)-df['rank'])/max(len(df)-1,1)

    total=len(df); eris_mask=(df['label']==args.eris_label).values
    
    def frac_in_top(pct):
        if eris_mask.sum() == 0: return 0.0
        k=math.ceil((pct/100.0)*total); top_mask=df['rank'].values<=k
        return 100.0*(eris_mask & top_mask).sum()/eris_mask.sum()
        
    stats={
        'top_5_pct':frac_in_top(5.0),'top_10_pct':frac_in_top(10.0),'top_15_pct':frac_in_top(15.0),'top_20_pct':frac_in_top(20.0),
        'total_candidates':int(total),'eris_candidates':int(eris_mask.sum()),'per_source_counts':{k:int(v) for k,v in per_source_counts.items()},
        'model':args.model,'pll_model':pll_model_name,'pll_mode':args.pll_mode,'pll_max_bsz':args.pll_max_bsz,
        'pll_random_k':args.pll_random_k,'pll_stride':args.pll_stride,'pll_first_n':args.pll_first_n,
        'amp':bool(args.amp),'compile':bool(args.compile),'ref_center':args.ref_center,'cscs_formula':args.cscs_formula,
        'semantic_change_mode': args.semantic_change_mode,
        'model_ckpt': args.model_ckpt, 'pll_model_ckpt': args.pll_model_ckpt,
        'ref_variant': args.ref_variant, 'target_variants': args.target_variant
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0]+'.summary.json','w') as f: json.dump(stats,f,indent=2)
    print(json.dumps(stats, indent=2))
    print('Loaded per source:', {k:int(v) for k,v in per_source_counts.items()}, 'Total Loaded:', len(seqs), 'Total Ranked:', len(df))

if __name__=='__main__':
    main()
