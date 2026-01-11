import os
import uuid
import shutil
import logging
import warnings
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.util import SiameseDistanceMetric
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Bio import SeqIO

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='CoV-SNN Contrastive Training (Experimental Data)')
    parser.add_argument('--loss_name', type=str, default="ContrastiveLoss", help='Loss function name')
    parser.add_argument('--pooling_mode', type=str, default="max", help='Pooling mode')
    parser.add_argument('--concat', type=str, default=None, help='Concatenation type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument("--wd", type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--projection_ratio', type=float, default=0.0, help='projection_ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin value')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='Warmup steps')
    parser.add_argument('--optimize_for_zero_shot', action='store_true', help='Save best model based on zero-shot performance')
    parser.add_argument('--distance_metric', type=str, default="cosine", choices=["euclidean", "cosine"], help='Distance metric to use')
    parser.add_argument('--normalize_embeddings', action='store_true', help='Normalize embeddings')
    parser.add_argument('--permutation_steps', type=int, default=1000, help='Number of permutation steps for p-value calculation')
    parser.add_argument('--bonferroni_correction', type=float, default=1.0, help='Bonferroni correction factor for p-value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def build_model(constants):
    # Base Transformer
    encoder = models.Transformer(
        model_name_or_path="./mlm_checkpoints/CoV-RoBERTa_2048",
        max_seq_length=1280,
        tokenizer_name_or_path="trained_tokenizer/"
    )
    
    dim = encoder.get_word_embedding_dimension() # 768
    
    # Pooling
    pooler = models.Pooling(dim, pooling_mode=constants["pooling"])
    
    modules = [encoder, pooler]
    
    # Optional Projection Layer with ReLU activation
    if constants["projection_ratio"] > 0:
        dense = models.Dense(
            in_features=dim, 
            out_features=int(dim * constants["projection_ratio"]), 
            activation_function=nn.ReLU()
        )
        modules.append(dense)
    
    # Optional Dropout
    if constants["dropout"] > 0:
        dropout = models.Dropout(constants["dropout"])
        modules.append(dropout)
        
    model = SentenceTransformer(modules=modules)
    return model

def prepare_data(constants):
    # Load sequences
    # Using relative paths assuming script is run from project root
    print("Loading data...")
    sig_seq = pd.read_csv('exp_data/sig_train_val_extended.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
    non_sig_seq = pd.read_csv('exp_data/non_sig_train_val_extended.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
    
    print(f"Sig (Negatives): {len(sig_seq)}, Non-Sig (Positives): {len(non_sig_seq)}")
    
    wt = str(SeqIO.read('exp_data/wild_type.fasta', 'fasta').seq)
    
    examples = []
    
    # Create pairs: [WT, Variant]
    # Label 0: Significant (different from WT -> Negative in this context effectively, or rather 'class 0')
    
    for neg in sig_seq:
        examples.append(InputExample(texts=[wt, neg], label=0))
    
    for pos in non_sig_seq:
        examples.append(InputExample(texts=[wt, pos], label=1))
        
    print(f"Total examples: {len(examples)}")
    
    # Shuffle and Split
    random.shuffle(examples)
    n = len(examples)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:train_size + val_size]
    test_examples = examples[train_size + val_size:]
    
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
    
    return train_examples, val_examples, test_examples

def prepare_specific_zero_shot_data(dataset_name):
    print(f"Loading Zero-Shot data for {dataset_name}...")
    wt = str(SeqIO.read('exp_data/wild_type.fasta', 'fasta').seq)
    
    if dataset_name == "greaney":
        non_sig_seq = pd.read_csv('exp_data/non_sig_seq_greaney_filtered.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
        sig_seq = pd.read_csv('exp_data/sig_seq_greaney.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
    elif dataset_name == "baum":
        non_sig_seq = pd.read_csv('exp_data/non_sig_seq_baum_filtered.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
        sig_seq = pd.read_csv('exp_data/sig_seq_baum.csv', header=None, names=['mutation', 'sequence'])['sequence'].tolist()
        num_non_sig_needed = len(sig_seq) * 20
        if len(non_sig_seq) >= num_non_sig_needed:
            non_sig_seq = random.sample(non_sig_seq, num_non_sig_needed)
        else:
            print(f"Warning: Not enough non-sig sequences for Baum set (Requested {num_non_sig_needed}, available {len(non_sig_seq)})")
    else:
         raise ValueError(f"Unknown negative set: {dataset_name}")

    print(f"Zero-Shot ({dataset_name}) -> Non-Sig (Pos): {len(non_sig_seq)}, Sig (Neg): {len(sig_seq)}")

    zero_test_examples = []

    # Non-significant -> Positive (Label 1)
    for seq in non_sig_seq:
        zero_test_examples.append(InputExample(texts=[wt, seq], label=1))

    # Significant -> Negative (Label 0)
    for seq in sig_seq:
        zero_test_examples.append(InputExample(texts=[wt, seq], label=0))

    random.shuffle(zero_test_examples)
    print(f"Zero-shot test set length for {dataset_name}: {len(zero_test_examples)}")
    
    return zero_test_examples

def plot_learning_curves(stats_dir, train_stats, eval_stats, test_stats, baum_stats, grny_stats):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    axs[0].plot(train_stats["epoch"], train_stats["loss"], label="Training")
    axs[0].plot(eval_stats["epoch"], eval_stats["loss"], label="Validation")
    axs[0].plot(test_stats["epoch"], test_stats["loss"], label="Test")
    axs[0].plot(baum_stats["epoch"], baum_stats["loss"], label="Baum")
    axs[0].plot(grny_stats["epoch"], grny_stats["loss"], label="Greaney")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Accuracy Plot
    axs[1].plot(train_stats["epoch"], train_stats["accuracy"], label="Training")
    axs[1].plot(eval_stats["epoch"], eval_stats["accuracy"], label="Validation")
    axs[1].plot(test_stats["epoch"], test_stats["accuracy"], label="Test")
    axs[1].plot(baum_stats["epoch"], baum_stats["accuracy"], label="Baum")
    axs[1].plot(grny_stats["epoch"], grny_stats["accuracy"], label="Greaney")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "learning_curves.pdf"))

def main():
    
    args = get_args()
    set_seed(args.seed)
    
    CONSTANTS = {
        "loss": args.loss_name,
        "negative_set": args.neg_set,
        "pooling": args.pooling_mode,
        "concat": args.concat,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "projection_ratio": args.projection_ratio,
        "dropout": args.dropout,
        "margin": args.margin,
        "distance_metric": args.distance_metric,
        "normalize_embeddings": args.normalize_embeddings,
        "permutation_steps": args.permutation_steps,
        "bonferroni_correction": args.bonferroni_correction,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed
    }
    
    # Print Constants
    for k, v in CONSTANTS.items():
        print(f"{k}: {v}")

    # Build Model
    model = build_model(CONSTANTS)

    # Prepare Data
    train_examples, val_examples, test_examples = prepare_data(CONSTANTS)
    
    # Prepare both zero shot datasets
    baum_examples = prepare_specific_zero_shot_data("baum")
    grny_examples = prepare_specific_zero_shot_data("greaney")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONSTANTS["batch_size"])

    # Define Loss
    metric = SiameseDistanceMetric.EUCLIDEAN if CONSTANTS["distance_metric"] == "euclidean" else SiameseDistanceMetric.COSINE

    if CONSTANTS["loss"] == "ContrastiveLoss":
        train_loss = losses.ContrastiveLoss(
            model=model,
            distance_metric=metric,
            margin=CONSTANTS["margin"]
        )
    elif CONSTANTS["loss"] == "OnlineContrastiveLoss":
        train_loss = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=metric,
            margin=CONSTANTS["margin"]
        )

    # Evaluators
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[e.texts[0] for e in val_examples],
        sentences2=[e.texts[1] for e in val_examples],
        labels=[e.label for e in val_examples],
        distance_metric=metric,
        normalize_embeddings=CONSTANTS["normalize_embeddings"],
        batch_size=CONSTANTS["batch_size"],
        margin=CONSTANTS["margin"],
        show_progress_bar=False,
        write_csv=True,
        name='Eval',
        permutation_steps=CONSTANTS["permutation_steps"],
        bonferroni_correction=CONSTANTS["bonferroni_correction"]
    )

    test_evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[e.texts[0] for e in test_examples],
        sentences2=[e.texts[1] for e in test_examples],
        labels=[e.label for e in test_examples],
        distance_metric=metric,
        normalize_embeddings=CONSTANTS["normalize_embeddings"],
        batch_size=CONSTANTS['batch_size'],
        margin=CONSTANTS['margin'],
        show_progress_bar=False,
        name="Test",
        permutation_steps=CONSTANTS["permutation_steps"],
        bonferroni_correction=CONSTANTS["bonferroni_correction"]
    )

    baum_evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[e.texts[0] for e in baum_examples],
        sentences2=[e.texts[1] for e in baum_examples],
        labels=[e.label for e in baum_examples],
        batch_size=CONSTANTS['batch_size'],
        margin=CONSTANTS['margin'],
        show_progress_bar=False,
        name="Baum",
        distance_metric=metric,
        normalize_embeddings=CONSTANTS["normalize_embeddings"],
        permutation_steps=CONSTANTS["permutation_steps"],
        bonferroni_correction=CONSTANTS["bonferroni_correction"]
    )

    grny_evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[e.texts[0] for e in grny_examples],
        sentences2=[e.texts[1] for e in grny_examples],
        labels=[e.label for e in grny_examples],
        batch_size=CONSTANTS['batch_size'],
        margin=CONSTANTS['margin'],
        show_progress_bar=False,
        name="Grny",
        distance_metric=metric,
        normalize_embeddings=CONSTANTS["normalize_embeddings"],
        permutation_steps=CONSTANTS["permutation_steps"],
        bonferroni_correction=CONSTANTS["bonferroni_correction"]
    )
    
    # Initialize list of evaluators
    zero_shot_evaluators = []
    
    # Determine order and which one is primary
    # If optimizing for zero shot, we might want to be specific about which one is the "main" score.
    # SequentialEvaluator uses the last one's score by default.
    # We will put the one matching args.neg_set last, just in case.
    if args.neg_set == "baum":
        zero_shot_evaluators = [grny_evaluator, baum_evaluator]
    else:
        zero_shot_evaluators = [baum_evaluator, grny_evaluator]
        
    zero_test_evaluator = SequentialEvaluator(zero_shot_evaluators, main_score_function=lambda scores: scores[-1])

    # Determine primary evaluator for model saving
    primary_evaluator = evaluator
    if args.optimize_for_zero_shot:
        print(f"Optimizing for Zero-Shot performance! (Primary target: {args.neg_set})")
        primary_evaluator = zero_test_evaluator

    # Output Directory
    output_dir = f"./outputs_exp_multi_eval/Exp_MultiZero_{CONSTANTS['loss']}_" \
                 f"PR{CONSTANTS['projection_ratio']}_" \
                 f"DR{CONSTANTS['dropout']}_" \
                 f"LR{CONSTANTS['learning_rate']}_" \
                 f"WD{CONSTANTS['weight_decay']}_" \
                 f"MR{CONSTANTS['margin']}_" \
                 f"DM{CONSTANTS['distance_metric']}_" \
                 f"SEED{CONSTANTS['seed']}"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    checkpoint_dir = f"{output_dir}/checkpoints"
    stats_dir = f"{output_dir}/stats"

    for d in [checkpoint_dir, stats_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Calculate warmup steps
    train_steps = len(train_dataloader) * CONSTANTS['epochs']
    print(f"Total training steps: {train_steps}, Warmup steps: {CONSTANTS['warmup_steps']}")

    class LoggingOptimizer(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0

        def step(self, closure=None):
            loss = super().step(closure)
            self.step_count += 1
            if self.step_count % len(train_dataloader) == 0:
                print(f"Learning Rate: {self.param_groups[-1]['lr']}")
            return loss

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=primary_evaluator,
        tester=test_evaluator,
        zero_shot_tester=zero_test_evaluator,
        epochs=CONSTANTS['epochs'],
        early_stopping=args.early_stopping,
        warmup_steps=CONSTANTS['warmup_steps'],
        optimizer_class=LoggingOptimizer,
        optimizer_params={'lr': CONSTANTS['learning_rate']},
        weight_decay=CONSTANTS['weight_decay'],
        output_path=output_dir,
        save_best_model=False,
        save_ckpt_on_eval=False,
        show_progress_bar=False,
        loss_name=CONSTANTS['loss']
    )

    # Save summary stats
    f_train_stats = os.path.join(stats_dir, 'Train.csv')
    f_eval_stats = os.path.join(stats_dir, 'Eval.csv')
    f_test_stats = os.path.join(stats_dir, 'Test.csv')
    f_baum_stats = os.path.join(stats_dir, 'Baum.csv') # Evaluator name='Baum'
    f_grny_stats = os.path.join(stats_dir, 'Grny.csv') # Evaluator name='Grny'

    train_stats = pd.read_csv(f_train_stats)
    eval_stats = pd.read_csv(f_eval_stats)
    test_stats = pd.read_csv(f_test_stats)
    baum_stats = pd.read_csv(f_baum_stats)
    grny_stats = pd.read_csv(f_grny_stats)

    summary_stats_header = ["Loss Name", "Neg Set", "Projection", "Dropout", "Learning Rate", "Weight Decay", "Margin", "Distance Metric",
                            "Epoch", "Train Loss", "Train Acc", "Train AUC",
                            "Eval Loss", "Eval Acc", "Eval Acc Threshold", "Eval AUC", "Eval AUC CI Lower", "Eval AUC CI Upper", "Eval AUC P-Value",
                            "Test Loss", "Test Acc", "Test Acc Threshold", "Test AUC", "Test AUC CI Lower", "Test AUC CI Upper", "Test AUC P-Value",
                            "Baum Loss", "Baum Acc", "Baum Acc Threshold", "Baum AUC", "Baum AUC CI Lower", "Baum AUC CI Upper", "Baum AUC P-Value",
                            "Grny Loss", "Grny Acc", "Grny Acc Threshold", "Grny AUC", "Grny AUC CI Lower", "Grny AUC CI Upper", "Grny AUC P-Value"]

    summary_stats = pd.DataFrame(columns=summary_stats_header)
    # Read for each Epoch and append to summary_stats
    for epoch in range(1, CONSTANTS['epochs'] + 1):
        summary_stats = summary_stats.append({
            "Loss Name": CONSTANTS['loss'],
            "Neg Set": CONSTANTS['negative_set'],
            "Projection Ratio": CONSTANTS['projection_ratio'],
            "Dropout": CONSTANTS['dropout'],
            "Learning Rate": CONSTANTS['learning_rate'],
            "Weight Decay": CONSTANTS['weight_decay'],
            "Margin": CONSTANTS['margin'],
            "Distance Metric": CONSTANTS['distance_metric'],
            "Epoch": epoch,
            "Train Loss": train_stats['loss'][epoch - 1],
            "Train Acc": train_stats['accuracy'][epoch - 1],
            "Train AUC": train_stats['auc'][epoch - 1],
            "Eval Loss": eval_stats['loss'][epoch - 1],
            "Eval Acc": eval_stats['accuracy'][epoch - 1],
            "Eval Acc Threshold": eval_stats['accuracy_thr'][epoch - 1],
            "Eval AUC": eval_stats['auc'][epoch - 1],
            "Eval AUC CI Lower": eval_stats['auc_ci_low'][epoch - 1],
            "Eval AUC CI Upper": eval_stats['auc_ci_high'][epoch - 1],
            "Eval AUC P-Value": eval_stats['auc_pval'][epoch - 1],
            "Test Loss": test_stats['loss'][epoch - 1],
            "Test Acc": test_stats['accuracy'][epoch - 1],
            "Test Acc Threshold": test_stats['accuracy_thr'][epoch - 1],
            "Test AUC": test_stats['auc'][epoch - 1],
            "Test AUC CI Lower": test_stats['auc_ci_low'][epoch - 1],
            "Test AUC CI Upper": test_stats['auc_ci_high'][epoch - 1],
            "Test AUC P-Value": test_stats['auc_pval'][epoch - 1],
            "Baum Loss": baum_stats['loss'][epoch - 1],
            "Baum Acc": baum_stats['accuracy'][epoch - 1],
            "Baum Acc Threshold": baum_stats['accuracy_thr'][epoch - 1],
            "Baum AUC": baum_stats['auc'][epoch - 1],
            "Baum AUC CI Lower": baum_stats['auc_ci_low'][epoch - 1],
            "Baum AUC CI Upper": baum_stats['auc_ci_high'][epoch - 1],
            "Baum AUC P-Value": baum_stats['auc_pval'][epoch - 1],
            "Grny Loss": grny_stats['loss'][epoch - 1],
            "Grny Acc": grny_stats['accuracy'][epoch - 1],
            "Grny Acc Threshold": grny_stats['accuracy_thr'][epoch - 1],
            "Grny AUC": grny_stats['auc'][epoch - 1],
            "Grny AUC CI Lower": grny_stats['auc_ci_low'][epoch - 1],
            "Grny AUC CI Upper": grny_stats['auc_ci_high'][epoch - 1],
            "Grny AUC P-Value": grny_stats['auc_pval'][epoch - 1]
        }, ignore_index=True)

    summary_stats.to_csv(os.path.join(stats_dir, 'Summary.csv'), index=False)


    # Print AUC values for epoch with best Test AUC
    best_test_auc_epoch = test_stats['auc'].idxmax()
    print(f"Best Test AUC: {test_stats['auc'][best_test_auc_epoch]} at Epoch: {best_test_auc_epoch} "
          f"(Baum AUC: {baum_stats['auc'][best_test_auc_epoch]}, Grny AUC: {grny_stats['auc'][best_test_auc_epoch]})")

    # Print AUC values for epoch with best Baum AUC
    best_baum_auc_epoch = baum_stats['auc'].idxmax()
    print(f"Best Baum AUC: {baum_stats['auc'][best_baum_auc_epoch]} at Epoch: {best_baum_auc_epoch} (Test AUC: {test_stats['auc'][best_baum_auc_epoch]})")

    # Print AUC values for epoch with best Grny AUC
    best_grny_auc_epoch = grny_stats['auc'].idxmax()
    print(f"Best Grny AUC: {grny_stats['auc'][best_grny_auc_epoch]} at Epoch: {best_grny_auc_epoch} (Test AUC: {test_stats['auc'][best_grny_auc_epoch]})")

    # Plot Learning Curves
    plot_learning_curves(stats_dir, train_stats, eval_stats, test_stats, baum_stats, grny_stats)

if __name__ == "__main__":
    main()
