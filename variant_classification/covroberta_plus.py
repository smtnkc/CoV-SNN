import os
import time
import warnings
import logging
from transformers import logging as hf_logging

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors
hf_logging.set_verbosity_error()

import argparse
import random
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from transformers import (
    RobertaTokenizerFast,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from sentence_transformers.util import SiameseDistanceMetric
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train CoV-SNN Classifier (Nested CV: Contrastive > Multiclass)")
parser.add_argument("--max_length", type=int, default=2048, choices=[128, 2048], help="Max sequence length (128 or 2048)")
parser.add_argument("--epochs", type=int, default=3, help="Epochs for Classification")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Classification")
parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps for Classification")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for Classification")
parser.add_argument("--contrastive_epochs", type=int, default=10, help="Epochs for contrastive pre-training")
parser.add_argument("--contrastive_batch_size", type=int, default=32, help="Batch size for contrastive pre-training")
parser.add_argument("--skip_contrastive", action="store_true", help="Skip contrastive stage (Debug only, reuses old models if found)")
parser.add_argument("--contrastive_lr", type=float, default=2e-5, help="Learning rate for contrastive pre-training")
parser.add_argument("--variants", nargs='+', default=["Alpha", "Beta", "Gamma", "Delta", "Omicron"],
                    help="Variants to include (default: all)")
parser.add_argument("--contrastive_strategy", default="all_vs_all", choices=["all_vs_all", "omicron_vs_others"],
                    help="Strategy for contrastive pair generation")
args = parser.parse_args()

# --- Configuration ---
MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size
GRAD_ACC = args.grad_acc
LR = args.lr

MODEL_PATH = f"../mlm_checkpoints/CoV-RoBERTa_{MAX_LENGTH}"
OUTPUT_DIR = f"../outputs/variant_classification/covroberta_plus"
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
    "Omicron": "unique_Omicron_2k.csv"
}

# Filter based on args
VARIANTS = args.variants
FILES = {v: ALL_FILES[v] for v in VARIANTS if v in ALL_FILES}
logger.info(f"Selected Variants: {VARIANTS}")

# Dynamic Label Mapping
LABEL2ID = {v: i for i, v in enumerate(VARIANTS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
logger.info(f"Label Mapping: {LABEL2ID}")


# ==========================================
# MODEL DEFINITIONS
# ==========================================

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class VariantClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        # Load via SentenceTransformer to guarantee we get the updated weights correctly
        logger.info(f"Loading backbone via SentenceTransformer from {model_path}")
        sbert_model = SentenceTransformer(model_path)
        self.roberta = sbert_model[0].auto_model
        
        self.config = self.roberta.config
        self.config.num_labels = num_labels
        self.config.id2label = ID2LABEL
        self.config.label2id = LABEL2ID
        
        # Use standard RoBERTa classification head
        self.classifier = RobertaClassificationHead(self.config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        sequence_output = outputs.last_hidden_state
        
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            log_output = {k: v for k, v in logs.items()}
            logging.getLogger(__name__).info(str(log_output))

# ==========================================
# DATA UTILS
# ==========================================

def load_all_data():
    """Loads all data into a single DataFrame with variant labels."""
    dfs = []
    for variant, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'sequence' in df.columns:
                df = df[['sequence']].copy()
                df['label'] = LABEL2ID[variant]
                df['variant'] = variant # Keep track of variant name for pair generation
                dfs.append(df)
            else:
                logger.error(f"Err: 'sequence' missing in {filename}")
        else:
            logger.warning(f"File {filepath} not found.")

    return pd.concat(dfs, ignore_index=True)

# ==========================================
# STEP 1: CONTRASTIVE PRETRAINING
# ==========================================

def generate_pairs(df, desc="Training"):
    examples = []
    
    # Organize sequences by variant
    all_seqs = {v: [] for v in VARIANTS}
    for _, row in df.iterrows():
        all_seqs[row['variant']].append(row['sequence'])
    
    logger.info(f"Generating contrastive pairs ({desc}) using strategy: {args.contrastive_strategy}")
    
    if args.contrastive_strategy == "all_vs_all":
        # All vs All logic
        r = 1
        done = False
        sequences = [all_seqs[v] for v in VARIANTS]
        
        # Safety check: ensure all variants have data
        if any(len(s) == 0 for s in sequences):
            logger.warning("One or more variants have NO data. 'all_vs_all' might fail or produce empty results.")
        
        while not done:
            for a_p_list_id in range(len(sequences)):
                # If this variant is empty, skip
                if len(sequences[a_p_list_id]) < 5:
                    continue

                n_list_ids = list(range(len(sequences))).copy()
                n_list_ids.remove(a_p_list_id)
                
                anchor_positive_list = sequences[a_p_list_id]
                anchor_positives = random.sample(anchor_positive_list, 5)
                anchor = anchor_positives[0]
                positives = anchor_positives[1:]
                anchor_positive_list.remove(anchor)
                
                for positive in positives:
                    examples.append(InputExample(texts=[anchor, positive], label=1))
                    
                for n_list_id in n_list_ids:
                    negative_list = sequences[n_list_id]
                    if len(negative_list) == 0: continue # Skip if no negatives for this variant
                    
                    negative = random.choice(negative_list)
                    negative_list.remove(negative)
                    examples.append(InputExample(texts=[anchor, negative], label=0))
                    
            if any(len(seqs) < 5 for seqs in sequences):
                done = True
                break
            r += 1

    elif args.contrastive_strategy == "omicron_vs_others":
        # Omicron vs Others logic
        if "Omicron" not in all_seqs or len(all_seqs["Omicron"]) == 0:
            logger.error("Strategy 'omicron_vs_others' selected but no Omicron sequences found!")
            return []
            
        omicron_sequences = all_seqs['Omicron']
        other_variants = [v for v in VARIANTS if v != 'Omicron']
        other_sequences_pool = []
        for v in other_variants:
            other_sequences_pool.extend(all_seqs[v])
            
        if not other_sequences_pool:
            logger.error("Strategy 'omicron_vs_others' selected but no 'Other' sequences found!")
            return []

        random.shuffle(omicron_sequences)
        
        for i in range(len(omicron_sequences)):
            anchor = omicron_sequences[i]

            possible_positives = [x for x in omicron_sequences if x != anchor]
            if len(possible_positives) < 4:
                positives = possible_positives
            else:
                positives = random.sample(possible_positives, 4)
                
            for pos in positives:
                examples.append(InputExample(texts=[anchor, pos], label=1))
                neg = random.choice(other_sequences_pool)
                examples.append(InputExample(texts=[anchor, neg], label=0))

    logger.info(f"Generated {len(examples)} pairs for {desc}.")
    return examples

def run_contrastive_step(train_df, fold_idx, output_dir):
    """
     trains the backbone on train_df using contrastive loss.
     Returns path to saved backbone.
    """
    fold_model_dir = os.path.join(output_dir, "contrastive_model")
    
    if args.skip_contrastive and os.path.exists(fold_model_dir):
        logger.info(f"Skipping contrastive step (found {fold_model_dir})")
        return fold_model_dir, 0.0, 0.0

    logger.info(f"Step 1: Contrastive Pretraining (Fold {fold_idx+1})")
    
    tokenizer_path = "/samet/CoV-SNN/trained_tokenizer"
    assert os.path.isdir(tokenizer_path), tokenizer_path

    word_embedding_model = models.Transformer(
        model_name_or_path=MODEL_PATH,
        max_seq_length=MAX_LENGTH,
        tokenizer_name_or_path=tokenizer_path
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), 
        pooling_mode='max')
        
    modules = [word_embedding_model, pooling_model]
    
    dim = word_embedding_model.get_word_embedding_dimension()
    dense_dim = int(dim * 0.2)
    dense = models.Dense(in_features=dim, out_features=dense_dim, activation_function=nn.ReLU())
    modules.append(dense)
    
    dropout = models.Dropout(0.2)
    modules.append(dropout)
    
    model = SentenceTransformer(modules=modules)

    # Generate Pairs from ALL Training Data of this Fold
    # We shuffle sequences first to ensure randomness in pair generation
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_examples = generate_pairs(train_df, desc="Full Fold Train")
    
    # Internal Split: 80% Train / 20% Val based on EXAMPLES (Pairs), not Sequences
    random.seed(42)
    random.shuffle(full_examples)
    
    cutoff = int(0.8 * len(full_examples))
    train_examples = full_examples[:cutoff]
    val_examples = full_examples[cutoff:]
    
    logger.info(f"Contrastive Split: {len(train_examples)} Train Pairs | {len(val_examples)} Val Pairs")
    
    # --- Prepare Folders ---
    checkpoint_dir = os.path.join(fold_model_dir, "checkpoints")
    stats_dir = os.path.join(fold_model_dir, "stats")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.contrastive_batch_size)
    train_loss = losses.OnlineContrastiveLoss(model=model,
        distance_metric=SiameseDistanceMetric.EUCLIDEAN,
        margin=2.0)
        
    # Evaluator for 'save_best_model'
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[ex.texts[0] for ex in val_examples],
        sentences2=[ex.texts[1] for ex in val_examples],
        labels=[ex.label for ex in val_examples],
        batch_size=args.contrastive_batch_size,
        distance_metric=SiameseDistanceMetric.EUCLIDEAN,
        margin=2.0,
        name='Eval',
        show_progress_bar=False,
        write_csv=True
    )
    
    logger.info("Training backbone...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.contrastive_epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params= {'lr': args.contrastive_lr},
        output_path=fold_model_dir,
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_save_total_limit=1
    )
    
    end_time = time.time()
    contrastive_time = end_time - start_time
    contrastive_gpu = 0.0
    if torch.cuda.is_available():
        contrastive_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    logger.info(f"Contrastive Training Time: {contrastive_time:.2f} s")
    logger.info(f"Contrastive Max GPU Memory: {contrastive_gpu:.2f} GB")
    
    # Cleanup
    del model
    del train_dataloader
    del evaluator
    torch.cuda.empty_cache()
    
    return os.path.join(fold_model_dir, "best_model"), contrastive_time, contrastive_gpu


# ==========================================
# STEP 2: MULTICLASS FINETUNING
# ==========================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    macro_f1 = precision_recall_fscore_support(labels, predictions, average='macro')[2]
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1,
        'f1_macro': macro_f1,
        'precision': precision,
        'recall': recall
    }

def run_classification_step(train_df, val_df, backbone_path, fold_idx, output_dir):
    logger.info(f"Step 2: Classification Finetuning (Fold {fold_idx+1})")
    
    # Tokenizer
    tokenizer_path = "/samet/CoV-SNN/trained_tokenizer"
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

    def tokenize_function(examples):
        # Sequences already space-joined at loading if needed
        return tokenizer(examples["sequence"], padding=False, truncation=True, max_length=MAX_LENGTH)

    logger.info("Loading datasets...")
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(val_df)

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True
    )
    eval_dataset = eval_dataset.map(
        tokenize_function, 
        batched=True
    )

    # Load Backbone from Step 1
    logger.info(f"Loading backbone via SentenceTransformer from {backbone_path}")
    model = VariantClassifier(model_path=backbone_path, num_labels=len(VARIANTS))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=GRAD_ACC,
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[LogCallback]
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    train_time = end_time - start_time
    train_gpu = 0.0
    if torch.cuda.is_available():
        train_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
    logger.info(f"Classification Training Time: {train_time:.2f} s")
    logger.info(f"Classification Max GPU Memory: {train_gpu:.2f} GB")

    # Save final model for this fold
    trainer.save_model(output_dir)
    return trainer, eval_dataset, train_time, train_gpu

# ==========================================
# STEP 3: EVALUATION
# ==========================================

def run_evaluation_step(trainer, eval_dataset, fold_idx):
    logger.info(f"Step 3: Evaluation (Fold {fold_idx+1})")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # Predict
    preds_output = trainer.predict(eval_dataset)
    
    end_time = time.time()
    eval_time = end_time - start_time
    eval_gpu = 0.0
    if torch.cuda.is_available():
        eval_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
    metrics = preds_output.metrics
    metrics['eval_time'] = eval_time
    metrics['eval_gpu'] = eval_gpu
    
    logger.info(f"Evaluation Time: {eval_time:.2f} s")
    logger.info(f"Evaluation Max GPU Memory: {eval_gpu:.2f} GB")
    
    # Confusion Matrix
    logits = preds_output.predictions
    pred_labels = np.argmax(logits, axis=1)
    true_labels = preds_output.label_ids
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Per-Class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
    
    logger.info(f"Fold {fold_idx+1} Confusion Matrix:\n{cm}")
    logger.info(f"Fold {fold_idx+1} Per-Class F1 (Alpha, Beta, Gamma, Delta, Omicron): {f1_per_class}")
    logger.info(f"Fold {fold_idx+1} Best F1: {metrics['f1']} Best accuracy: {metrics['accuracy']}")
    
    # Add to metrics dict
    metrics['confusion_matrix'] = cm.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()
    
    return metrics

# ==========================================
# MAIN LOOP
# ==========================================

def main():
    logger.info("Starting Nested Cross-Validation...")
    
    # 0. Load Data
    df = load_all_data()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        logger.info(f"{'#'*40}\nFOLD {fold+1} / 5\n{'#'*40}")
        
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 1. Split
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        logger.info(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
        
        # 2. Step 1: Contrastive Pretraining (On Train Split)
        backbone_path, c_time, c_gpu = run_contrastive_step(train_df, fold, fold_dir)
        
        # 3. Step 2: Supervised Finetuning (On Train Split)
        # We start fresh classifier training using the NEW backbone
        trainer, eval_dataset, clf_time, clf_gpu = run_classification_step(
            train_df, val_df, 
            os.path.join(backbone_path), fold, fold_dir)
        
        # 4. Step 3: Evaluation (On Val Split)
        metrics = run_evaluation_step(trainer, eval_dataset, fold)
        
        # Add telemetry to metrics dict for aggregation
        metrics['contrastive_train_time'] = c_time
        metrics['contrastive_train_gpu'] = c_gpu
        metrics['classifier_train_time'] = clf_time
        metrics['classifier_train_gpu'] = clf_gpu
        
        fold_results.append(metrics)
        
        # Cleanup fold artifacts to save space (Optional, keep for debugging if needed)
        # shutil.rmtree(os.path.join(fold_dir, "contrastive_model")) 

    # Final Summary
    logger.info(f"\n{'='*20} Final Results {'='*20}")
    avg_acc = np.mean([m['test_accuracy'] for m in fold_results]) # Note: 'test_accuracy' usually key from predict
    avg_macro_f1 = np.mean([m['test_f1_macro'] for m in fold_results])
    
    # Telemetry Averages
    avg_c_time = np.mean([m['contrastive_train_time'] for m in fold_results])
    avg_c_gpu = np.mean([m['contrastive_train_gpu'] for m in fold_results])
    avg_clf_time = np.mean([m['classifier_train_time'] for m in fold_results])
    avg_clf_gpu = np.mean([m['classifier_train_gpu'] for m in fold_results])
    avg_eval_time = np.mean([m['eval_time'] for m in fold_results])
    avg_eval_gpu = np.mean([m['eval_gpu'] for m in fold_results])

    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Average Macro F1: {avg_macro_f1:.4f}")
    logger.info(f"{'-'*20}")
    logger.info(f"Avg Contrastive Train Time: {avg_c_time:.2f} s | GPU: {avg_c_gpu:.2f} GB")
    logger.info(f"Avg Classifier Train Time: {avg_clf_time:.2f} s | GPU: {avg_clf_gpu:.2f} GB")
    logger.info(f"Avg Evaluation Time: {avg_eval_time:.2f} s | GPU: {avg_eval_gpu:.2f} GB")

if __name__ == "__main__":
    main()
