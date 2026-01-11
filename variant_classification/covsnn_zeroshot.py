import os
import logging
import argparse
import random
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from sentence_transformers.util import SiameseDistanceMetric
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train CoV-SNN Contrastive (Zero-Shot)")
parser.add_argument("--max_length", type=int, default=2048, choices=[128, 2048], help="Max sequence length")

# Contrastive Hyperparams
parser.add_argument("--contrastive_epochs", type=int, default=10, help="Epochs for contrastive pre-training")
parser.add_argument("--contrastive_batch_size", type=int, default=32, help="Batch size for contrastive pre-training")
parser.add_argument("--contrastive_lr", type=float, default=1e-4, help="Learning rate for contrastive pre-training")

parser.add_argument("--training_strategy", default="omicron_vs_others", 
                    choices=["omicron_vs_others", "omicron_vs_delta"],
                    help="Strategy for contrastive pair generation")
parser.add_argument("--test_strategy", default="new_vs_eris", choices=["new_vs_eris"],
                    help="Strategy for zero-shot testing")

args = parser.parse_args()

# --- Configuration ---
MAX_LENGTH = args.max_length
MODEL_PATH = f"../mlm_checkpoints/CoV-RoBERTa_{MAX_LENGTH}"
OUTPUT_DIR = f"../outputs/variant_classification/covsnn_zeroshot_{args.training_strategy}_output"
DATA_DIR = "../data"

# Setup Logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dataset filenames
ALL_FILES = {
    "Alpha": "unique_Alpha_2k.csv",
    "Beta": "unique_Beta_2k.csv",
    "Gamma": "unique_Gamma_2k.csv",
    "Delta": "unique_Delta_2k.csv",
    "Omicron": "unique_Omicron_2k.csv",
    "New": "unique_New_2k.csv",
    "Eris": "unique_Eris_2k.csv"
}

# Define which files are for training vs testing
TRAIN_VARIANTS = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]
TEST_VARIANTS = ["New", "Eris"]

# Dynamic Label Mapping (Just for reference, though we don't do classification finetuning)
# We keep the training variants in the main map
LABEL2ID = {v: i for i, v in enumerate(TRAIN_VARIANTS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ==========================================
# DATA UTILS
# ==========================================

def load_data(variants):
    """Loads specific variants into a single DataFrame."""
    dfs = []
    for variant in variants:
        filename = ALL_FILES.get(variant)
        if not filename:
            logger.warning(f"Variant {variant} not defined in file list.")
            continue
            
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'sequence' in df.columns:
                df = df[['sequence']].copy()
                df['variant'] = variant
                dfs.append(df)
            else:
                logger.error(f"Err: 'sequence' missing in {filename}")
        else:
            logger.warning(f"File {filepath} not found.")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ==========================================
# PAIR GENERATION
# ==========================================

def generate_training_pairs(df):
    examples = []
    
    # Organize sequences by variant
    # We only care about TRAIN_VARIANTS here
    all_seqs = {v: [] for v in TRAIN_VARIANTS if v in df['variant'].unique()}
    for _, row in df.iterrows():
        if row['variant'] in all_seqs:
            all_seqs[row['variant']].append(row['sequence'])
    
    strategy = args.training_strategy
    logger.info(f"Generating training pairs using strategy: {strategy}")
    
    if strategy == "omicron_vs_others":
        if "Omicron" not in all_seqs or not all_seqs["Omicron"]:
            logger.error("Strategy omicron_vs_others: No Omicron data found.")
            return []
            
        omicron_sequences = all_seqs["Omicron"]  # User snippet: omicron_sequences
        
        # others = [alpha, beta, delta, gamma]
        # We need to construct this list carefully from available variants in TRAIN_VARIANTS
        # The user snippet expects 4 specific variants: Alpha, Beta, Delta, Gamma
        # Let's try to fetch them if they exist
        target_others = ["Alpha", "Beta", "Delta", "Gamma"]
        others = []
        for v in target_others:
            if v in all_seqs:
                others.append(all_seqs[v])
            else:
                logger.warning(f"Variant {v} missing for omicron_vs_others strategy.")
        
        if not others:
            logger.error("No 'Other' variants found.")
            return []

        # Logic from user snippet
        # if CONSTANTS["NEG_SET"] == "other":
        for i, anc in enumerate(omicron_sequences):
            # get 4 random omicron sequences
            # Use distinct sampling if possible, else replacement
            if len(omicron_sequences) >= 4:
                positives = random.sample(omicron_sequences, 4)
            else:
                positives = random.choices(omicron_sequences, k=4)
                
            for p, pos in enumerate(positives):
                # neg = others[p][i]
                # We need to ensure p < len(others) and i < len(others[p]) to avoid crash
                # User provided: others = [alpha, beta, delta, gamma] (len 4)
                # positives has len 4. So p goes 0..3.
                # If we have fewer than 4 other variants, we must wrap p
                
                other_subset = others[p % len(others)]
                
                # Safety for i
                neg = other_subset[i % len(other_subset)]
                
                examples.append(InputExample(texts=[anc, pos], label=1))
                examples.append(InputExample(texts=[anc, neg], label=0))

    elif strategy == "omicron_vs_delta":
        # if CONSTANTS["NEG_SET"] == "delta":
        if "Omicron" not in all_seqs or "Delta" not in all_seqs:
            logger.error("Strategy omicron_vs_delta: Missing Omicron or Delta data.")
            return []
            
        omicron_sequences = all_seqs["Omicron"]
        delta_sequences = all_seqs["Delta"]
        
        for i, anc in enumerate(omicron_sequences):
            # get 4 random omicron sequences
            if len(omicron_sequences) >= 4:
                positives = random.sample(omicron_sequences, 4)
            else:
                positives = random.choices(omicron_sequences, k=4)
                
            # get 4 random delta sequences
            if len(delta_sequences) >= 4:
                negatives = random.sample(delta_sequences, 4)
            else:
                negatives = random.choices(delta_sequences, k=4)
                
            for pos, neg in zip(positives, negatives):
                examples.append(InputExample(texts=[anc, pos], label=1))
                examples.append(InputExample(texts=[anc, neg], label=0))

    logger.info(f"Generated {len(examples)} training pairs.")
    return examples

def generate_test_pairs(test_df):
    """
    Generates pairs for zero-shot testing (New vs Eris).
    Positives: New-New, Eris-Eris
    Negatives: New-Eris
    """
    examples = []
    
    unique_variants = test_df['variant'].unique()
    # Expecting ['New', 'Eris'] usually, but let's be generic based on input
    all_seqs = {v: test_df[test_df['variant'] == v]['sequence'].tolist() for v in unique_variants}
    
    logger.info(f"Generating TEST pairs for variants: {unique_variants}")
    
    # We want a balanced evaluation set. 
    # Let's generate N positive pairs and N negative pairs.
    
    # Check emptiness
    if any(len(seqs) < 2 for seqs in all_seqs.values()):
        logger.warning("Not enough sequences in test variants to form pairs properly.")
    
    # 1. Intra-class (Positives)
    # New-New
    if "New" in all_seqs:
        seqs = all_seqs["New"]
        # Generate, say, 1000 pairs or iterate all?
        # Let's iterate all to be deterministic-ish
        for i in range(len(seqs)):
            anchor = seqs[i]
            # Pick a random positive
            pos = random.choice(seqs)
            examples.append(InputExample(texts=[anchor, pos], label=1))
            
    # Eris-Eris
    if "Eris" in all_seqs:
        seqs = all_seqs["Eris"]
        for i in range(len(seqs)):
            anchor = seqs[i]
            pos = random.choice(seqs)
            examples.append(InputExample(texts=[anchor, pos], label=1))
            
    # 2. Inter-class (Negatives)
    # New vs Eris
    if "New" in all_seqs and "Eris" in all_seqs:
        new_seqs = all_seqs["New"]
        eris_seqs = all_seqs["Eris"]
        
        # We want roughly same amount of negatives as positives
        total_positives = len(examples)
        
        # Generate New-Eris pairs
        for i in range(total_positives):
            anchor = random.choice(new_seqs)
            neg = random.choice(eris_seqs)
            examples.append(InputExample(texts=[anchor, neg], label=0))
            
    logger.info(f"Generated {len(examples)} TEST pairs.")
    return examples


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_fold(fold_idx, train_df, val_df, test_df):
    logger.info(f"\n{'='*20} Fold {fold_idx+1} {'='*20}")
    
    # 1. Initialize Model for this fold (Fresh start)
    logger.info(f"Initializing SBERT from {MODEL_PATH}")
    word_embedding_model = models.Transformer(
        model_name_or_path=MODEL_PATH,
        max_seq_length=MAX_LENGTH,
        tokenizer_name_or_path='./tok')
        
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), 
        pooling_mode='max')
        
    modules = [word_embedding_model, pooling_model]
    
    # Projection Head
    dim = word_embedding_model.get_word_embedding_dimension()
    dense_dim = int(dim * 0.2)
    dense = models.Dense(in_features=dim, out_features=dense_dim, activation_function=nn.ReLU())
    modules.append(dense)
    modules.append(models.Dropout(0.2))
    
    model = SentenceTransformer(modules=modules)
    
    # 2. Prepare Training Data
    train_examples = generate_training_pairs(train_df)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.contrastive_batch_size)
    
    train_loss = losses.OnlineContrastiveLoss(model=model,
        distance_metric=SiameseDistanceMetric.EUCLIDEAN,
        margin=2.0)

    # 3. Prepare Evaluators (Before Training)
    logger.info("Generating Validation Pairs...")
    val_examples = generate_training_pairs(val_df)
    evaluator_val = None
    if val_examples:
        evaluator_val = evaluation.BinaryClassificationEvaluator(
            sentences1=[ex.texts[0] for ex in val_examples],
            sentences2=[ex.texts[1] for ex in val_examples],
            labels=[ex.label for ex in val_examples],
            batch_size=args.contrastive_batch_size,
            distance_metric=SiameseDistanceMetric.EUCLIDEAN,
            margin=2.0,
            show_progress_bar=False,
            name='Validation-Test'
        )

    logger.info("Generating Zero-Shot Test Pairs...")
    test_examples = generate_test_pairs(test_df)
    evaluator_test = None
    if test_examples:
        evaluator_test = evaluation.BinaryClassificationEvaluator(
            sentences1=[ex.texts[0] for ex in test_examples],
            sentences2=[ex.texts[1] for ex in test_examples],
            labels=[ex.label for ex in test_examples],
            batch_size=args.contrastive_batch_size,
            distance_metric=SiameseDistanceMetric.EUCLIDEAN,
            margin=2.0,
            show_progress_bar=False,
            name='ZeroShot-Test'
        )

    # 4. Train
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx+1}")
    os.makedirs(fold_output_dir, exist_ok=True)
    model_output_dir = os.path.join(fold_output_dir, "model")
    os.makedirs(model_output_dir, exist_ok=True)
    stats_dir = os.path.join(model_output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    logger.info(f"Starting Training for Fold {fold_idx+1}...")
    
    # Using kwargs to match train_test_contrastive.py pattern
    # Note: 'tester' arg in reference seems to map to 'test_evaluator' in my code context (ZeroShot is separate)
    # But here I have Validation (Evaluator) and ZeroShot (Test). 
    # Reference has: evaluator(Val), tester(Test), zero_shot_tester(Zero).
    # I will map: evaluator -> evaluator_val, zero_shot_tester -> evaluator_test.
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.contrastive_epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': args.contrastive_lr},
        output_path=model_output_dir,
        evaluator=evaluator_val,
        zero_shot_tester=evaluator_test, # Passing this as requested by user pattern
        show_progress_bar=True,
        save_best_model=True
    )
    
    # 5. Return Metrics (Read from stats if possible, or just what we have)
    metrics = {}
    # Since we rely on internal logging/CSV saving from the custom fit, we might not get direct returns easily 
    # without reading the CSVs like the reference script does.
    
    # Read CSVs for last epoch metrics
    try:
        val_csv = pd.read_csv(os.path.join(stats_dir, 'Validation-Test.csv')) # Name comes from evaluator name
        metrics['val_score'] = val_csv['accuracy'].iloc[-1] if not val_csv.empty else 0.0
    except:
        logger.warning("Could not read Validation-Test.csv")
        
    try:
        zero_csv = pd.read_csv(os.path.join(stats_dir, 'ZeroShot-Test.csv'))
        metrics['test_score'] = zero_csv['accuracy'].iloc[-1] if not zero_csv.empty else 0.0
    except:
        logger.warning("Could not read ZeroShot-Test.csv")

    logger.info(f"Fold {fold_idx+1} Metrics: {metrics}")
    return metrics

def main():
    logger.info("Starting Zero-Shot Contrastive Pipeline with 5-Fold CV...")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Available. Device: {torch.cuda.get_device_name(0)}")
    
    # 1. Load All Data
    train_full_df = load_data(TRAIN_VARIANTS)
    test_df = load_data(TEST_VARIANTS)
    
    if train_full_df.empty:
        logger.error("No training data found. Exiting.")
        return

    # 2. Stratified K-Fold
    # We stratify by Variant to ensure balanced variants in train/val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    # StratifiedKFold needs y labels to stratify
    # We can encode variants as integers for stratification
    variant_labels = [LABEL2ID[v] for v in train_full_df['variant']]
    
    for fold_idx, (train_index, val_index) in enumerate(skf.split(train_full_df, variant_labels)):
        train_df = train_full_df.iloc[train_index].copy()
        val_df = train_full_df.iloc[val_index].copy()
        
        logger.info(f"Fold {fold_idx+1}: Train Size {len(train_df)} | Val Size {len(val_df)}")
        
        metrics = run_fold(fold_idx, train_df, val_df, test_df)
        fold_results.append(metrics)
        
    # 3. Summary
    logger.info(f"\n{'='*20} Final Summary {'='*20}")
    
    avg_val = np.mean([m.get('val_score', 0.0) for m in fold_results])
    avg_test = np.mean([m.get('test_score', 0.0) for m in fold_results])
    
    logger.info(f"Average Validation Score: {avg_val:.4f}")
    logger.info(f"Average Zero-Shot Test Score: {avg_test:.4f}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
