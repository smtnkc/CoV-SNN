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
from sentence_transformers.util import SiameseDistanceMetric
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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
    parser = argparse.ArgumentParser(description='CoV-SNN Contrastive Training')
    parser.add_argument('--loss_name', type=str, default="OnlineContrastiveLoss", help='Loss function name')
    parser.add_argument('--neg_set', type=str, default="delta", help='Negative set type')
    parser.add_argument('--pooling_mode', type=str, default="max", help='Pooling mode')
    parser.add_argument('--concat', type=str, default=None, help='Concatenation type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument("--wd", type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--projection_ratio', type=float, default=0.0, help='Projection Ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin value')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--optimize_for_zero_shot', action='store_true', help='Save best model based on zero-shot performance')
    parser.add_argument('--zero_shot_eval_type', type=str, default="voting", choices=["voting", "binary"], help='Type of zero-shot evaluator: voting or binary')
    parser.add_argument('--gstats', type=str, default="global_stats_contrastive.csv", help='CSV file path')
    parser.add_argument('--normalize_embeddings', action='store_true', help='Normalize embeddings')
    parser.add_argument('--distance_metric', type=str, default="cosine", choices=["euclidean", "cosine"], help='Distance metric to use')
    parser.add_argument('--permutation_steps', type=int, default=1000, help='Number of permutation steps for p-value calculation')
    parser.add_argument('--bonferroni_correction', type=float, default=1.0, help='Bonferroni correction factor for p-value')
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

def split_data(sequences, train_ratio=0.7, val_ratio=0.1):
    # Split data into train, validation, and test sets (70% train, 10% val, 20% test)
    random.shuffle(sequences)
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]


def generate_dataset(anchors, others_pool_list, mode="delta", negatives_pool=None):
    dataset = []
    
    if mode == "other":
        # others_pool_list should be a list of list-like pools: [alphas, betas, deltas, gammas]
        for anc in anchors:
            positives = random.sample(anchors, 4)
            for p, pos in enumerate(positives):
                # Sample 1 negative from the specific variant pool corresponding to p
                neg_pool = others_pool_list[p]
                neg = random.choice(neg_pool)
                dataset.append(InputExample(texts=[anc, pos], label=1))
                dataset.append(InputExample(texts=[anc, neg], label=0))
                
    elif mode == "delta":
        for anc in anchors:
            positives = random.sample(anchors, 4)
            negatives = random.sample(negatives_pool, 4)
            for pos, neg in zip(positives, negatives):
                dataset.append(InputExample(texts=[anc, pos], label=1))
                dataset.append(InputExample(texts=[anc, neg], label=0))
                
    return dataset

def prepare_data(constants):
    # Load all sequences
    limit = 2000
    omicron_full = pd.read_csv("../data/unique_Omicron_2k.csv")["sequence"].tolist()[:limit]
    alpha_full = pd.read_csv("../data/unique_Alpha_2k.csv")["sequence"].tolist()[:limit]
    beta_full = pd.read_csv("../data/unique_Beta_2k.csv")["sequence"].tolist()[:limit]
    delta_full = pd.read_csv("../data/unique_Delta_2k.csv")["sequence"].tolist()[:limit]
    gamma_full = pd.read_csv("../data/unique_Gamma_2k.csv")["sequence"].tolist()[:limit]

    # Split each variant's sequences to avoid leakage
    o_train, o_val, o_test = split_data(omicron_full)
    a_train, a_val, a_test = split_data(alpha_full)
    b_train, b_val, b_test = split_data(beta_full)
    d_train, d_val, d_test = split_data(delta_full)
    g_train, g_val, g_test = split_data(gamma_full)

    train_examples = []
    val_examples = []
    test_examples = []

    if constants["negative_set"] == "other":
        others_train = [a_train, b_train, d_train, g_train]
        train_examples = generate_dataset(o_train, others_train, mode="other")
        
        others_val = [a_val, b_val, d_val, g_val]
        val_examples = generate_dataset(o_val, others_val, mode="other")
        
        others_test = [a_test, b_test, d_test, g_test]
        test_examples = generate_dataset(o_test, others_test, mode="other")
        
    elif constants["negative_set"] == "delta":
        train_examples = generate_dataset(o_train, None, mode="delta", negatives_pool=d_train)
        val_examples = generate_dataset(o_val, None, mode="delta", negatives_pool=d_val)
        test_examples = generate_dataset(o_test, None, mode="delta", negatives_pool=d_test)

    print(f"Training set: {len(train_examples)}, Val set: {len(val_examples)}, Test set: {len(test_examples)}")
    
    return train_examples, val_examples, test_examples

def prepare_zero_shot_data():
    # Load data for zero-shot testing
    limit = 2000
    o = pd.read_csv("../data/unique_Omicron_2k.csv")["sequence"].tolist()[:limit]
    e = pd.read_csv("../data/unique_Eris_2k.csv")["sequence"].tolist()[:limit] # Negatives
    n = pd.read_csv("../data/unique_New_2k.csv")["sequence"].tolist()[:limit]  # Positives

    # Shuffle all sequences
    random.shuffle(o)
    random.shuffle(e)
    random.shuffle(n)
    
    # Return lists directly
    return o, n, e

def plot_learning_curves(stats_dir, train_stats, eval_stats, test_stats, zero_stats):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    axs[0].plot(train_stats["epoch"], train_stats["loss"], label="Training")
    axs[0].plot(eval_stats["epoch"], eval_stats["loss"], label="Validation")
    axs[0].plot(test_stats["epoch"], test_stats["loss"], label="Test")
    axs[0].plot(zero_stats["epoch"], zero_stats["loss"], label="Zero-shot")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Accuracy Plot
    axs[1].plot(train_stats["epoch"], train_stats["accuracy"], label="Training")
    axs[1].plot(eval_stats["epoch"], eval_stats["accuracy"], label="Validation")
    axs[1].plot(test_stats["epoch"], test_stats["accuracy"], label="Test")
    axs[1].plot(zero_stats["epoch"], zero_stats["accuracy"], label="Zero-shot")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "learning_curves.pdf"))

def main():
    set_seed()
    args = get_args()
    
    CONSTANTS = {
        "VOC_NAMES": ["Alpha", "Beta", "Delta", "Gamma", "Omicron"],
        "loss": args.loss_name,
        "negative_set": args.neg_set,
        "zs_eval_type": args.zero_shot_eval_type,
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
        "bonferroni_correction": args.bonferroni_correction
    }
    
    # Print Constants
    for k, v in CONSTANTS.items():
        print(f"{k}: {v}")

    # Build Model
    model = build_model(CONSTANTS)
    # print(model)

    # Prepare Data
    train_examples, val_examples, test_examples = prepare_data(CONSTANTS)
    zero_anchors, zero_positives, zero_negatives = prepare_zero_shot_data()

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

    if CONSTANTS["zs_eval_type"] == "voting":
        zero_test_evaluator = evaluation.ZeroShotVotingEvaluator(
            anchors=zero_anchors,
            positives=zero_positives,
            negatives=zero_negatives,
            batch_size=CONSTANTS['batch_size'],
            name="Zero",
            distance_metric=metric,
            normalize_embeddings=CONSTANTS["normalize_embeddings"],
            show_progress_bar=False,
            margin=CONSTANTS["margin"],
            permutation_steps=CONSTANTS["permutation_steps"],
            bonferroni_correction=CONSTANTS["bonferroni_correction"]
        )
    else:
        # Prepare pairs for BinaryClassificationEvaluator
        z_s1 = []
        z_s2 = []
        z_labels = []
        for i in range(len(zero_anchors)):
            # Positive pair
            z_s1.append(zero_anchors[i])
            z_s2.append(zero_positives[i])
            z_labels.append(1)
            # Negative pair
            z_s1.append(zero_anchors[i])
            z_s2.append(zero_negatives[i])
            z_labels.append(0)
            
        zero_test_evaluator = evaluation.BinaryClassificationEvaluator(
            sentences1=z_s1,
            sentences2=z_s2,
            labels=z_labels,
            batch_size=CONSTANTS['batch_size'],
            normalize_embeddings=CONSTANTS["normalize_embeddings"],
            margin=CONSTANTS['margin'],
            show_progress_bar=False,
            name="Zero",
            distance_metric=metric,
            permutation_steps=CONSTANTS["permutation_steps"],
            bonferroni_correction=CONSTANTS["bonferroni_correction"]
        )

    # Determine primary evaluator for model saving
    primary_evaluator = evaluator
    if args.optimize_for_zero_shot:
        print("Optimizing for Zero-Shot performance!")
        primary_evaluator = zero_test_evaluator

    # Output Directory
    neg_set = "OD" if CONSTANTS['negative_set'] == "delta" else "OO"
    output_dir = f"./outputs_2026/{CONSTANTS['loss']}_{neg_set}_" \
                 f"PR{CONSTANTS['projection_ratio']}_" \
                 f"DR{CONSTANTS['dropout']}_" \
                 f"LR{CONSTANTS['learning_rate']}_" \
                 f"WD{CONSTANTS['weight_decay']}_" \
                 f"MR{CONSTANTS['margin']}_" \
                 f"DM{CONSTANTS['distance_metric']}"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        # print(f"Removed directory: {output_dir}")

    checkpoint_dir = f"{output_dir}/checkpoints"
    stats_dir = f"{output_dir}/stats"

    for d in [checkpoint_dir, stats_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            # print(f"Created directory: {d}")

    # Calculate warmup steps
    train_steps = len(train_dataloader) * CONSTANTS['epochs']
    warmup_steps = int(train_steps * args.warmup_ratio)
    print(f"Total training steps: {train_steps}, Warmup steps: {warmup_steps}")

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=primary_evaluator,
        tester=test_evaluator,
        zero_shot_tester=zero_test_evaluator,
        epochs=CONSTANTS['epochs'],
        early_stopping=args.early_stopping,
        warmup_steps=warmup_steps,
        optimizer_class=torch.optim.AdamW,
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
    f_zero_stats = os.path.join(stats_dir, 'Zero.csv')

    train_stats = pd.read_csv(f_train_stats)
    eval_stats = pd.read_csv(f_eval_stats)
    test_stats = pd.read_csv(f_test_stats)
    zero_stats = pd.read_csv(f_zero_stats)

    summary_stats_header = ["Loss Name", "Projection", "Dropout", "Learning Rate", "Weight Decay", "Margin", "Distance Metric",
                            "Epoch", "Train Loss", "Train Acc", "Train AUC",
                            "Eval Loss", "Eval Acc", "Eval Acc Threshold", "Eval AUC", "Eval AUC CI Lower", "Eval AUC CI Upper", "Eval AUC P-Value",
                            "Test Loss", "Test Acc", "Test Acc Threshold", "Test AUC", "Test AUC CI Lower", "Test AUC CI Upper", "Test AUC P-Value",
                            "Zero Loss", "Zero Acc", "Zero Acc Threshold", "Zero AUC", "Zero AUC CI Lower", "Zero AUC CI Upper", "Zero AUC P-Value"]

    summary_stats = pd.DataFrame(columns=summary_stats_header)
    # Read for each Epoch and append to summary_stats
    for epoch in range(1, CONSTANTS['epochs'] + 1):
        summary_stats = summary_stats.append({
            "Loss Name": CONSTANTS['loss'],
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
            "Zero Loss": zero_stats['loss'][epoch - 1],
            "Zero Acc": zero_stats['accuracy'][epoch - 1],
            "Zero Acc Threshold": zero_stats['accuracy_thr'][epoch - 1],
            "Zero AUC": zero_stats['auc'][epoch - 1],
            "Zero AUC CI Lower": zero_stats['auc_ci_low'][epoch - 1],
            "Zero AUC CI Upper": zero_stats['auc_ci_high'][epoch - 1],
            "Zero AUC P-Value": zero_stats['auc_pval'][epoch - 1]
        }, ignore_index=True)

    summary_stats.to_csv(os.path.join(stats_dir, 'Summary.csv'), index=False)


    # Print AUC values for epoch with best Test AUC
    best_test_auc_epoch = test_stats['auc'].idxmax()
    print(f"Best Test AUC: {test_stats['auc'][best_test_auc_epoch]} at Epoch: {best_test_auc_epoch} (Zero AUC: {zero_stats['auc'][best_test_auc_epoch]})")

    # Print AUC values for epoch with best Zero AUC
    best_zero_auc_epoch = zero_stats['auc'].idxmax()
    print(f"Best Zero AUC: {zero_stats['auc'][best_zero_auc_epoch]} at Epoch: {best_zero_auc_epoch} (Test AUC: {test_stats['auc'][best_zero_auc_epoch]})")

    # Plot Learning Curves
    plot_learning_curves(stats_dir, train_stats, eval_stats, test_stats, zero_stats)

if __name__ == "__main__":
    main()
