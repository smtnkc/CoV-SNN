import os
import time
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset

# --- Configuration ---
MODEL_PATH = "hunarbatra/CoVBERT"
BASE_MODEL_NAME = "hunarbatra/CoVBERT"
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs/variant_classification/covbert"
MAX_LENGTH = 2048
BATCH_SIZE = 4 

# Dataset filenames
FILES = {
    "Alpha": "unique_Alpha_2k.csv",
    "Beta": "unique_Beta_2k.csv",
    "Gamma": "unique_Gamma_2k.csv",
    "Delta": "unique_Delta_2k.csv",
    "Omicron": "unique_Omicron_2k.csv"
}

# Label Mapping
LABEL2ID = {
    "Alpha": 0,
    "Beta": 1,
    "Gamma": 2,
    "Delta": 3,
    "Omicron": 4
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_data():
    """Loads and concatenates data from CSV files."""
    dfs = []
    for variant, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            logging.info(f"Loading {variant} from {filepath}...")
            df = pd.read_csv(filepath)
            if 'sequence' not in df.columns:
                 logging.error(f"  Error: 'sequence' column missing in {filename}")
                 continue
            
            df = df[['sequence']].copy()
            df['label'] = LABEL2ID[variant]
            dfs.append(df)
        else:
            logging.warning(f"Warning: File {filepath} not found.")

    if not dfs:
        raise ValueError("No data loaded. Please check data directory.")

    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total sequences loaded: {len(combined_df)}")
    return combined_df

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            log_output = {k: v for k, v in logs.items()}
            logging.getLogger(__name__).info(str(log_output))

def main():
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 1. Load Data
    df = load_data()

    # 2. Cross-Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Initialize Tokenizer
    logger.info(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    # CoVBERT tokenization: standard (no spaces needed), do_lower_case=False
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, do_lower_case=False)

    def tokenize_function(examples):
        return tokenizer(examples["sequence"], padding=False, truncation=True, max_length=MAX_LENGTH)

    # Metrics storage
    fold_metrics = []
    train_times = []
    eval_times = []
    train_gpu_memory = []
    eval_gpu_memory = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        logger.info(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")
        
        # Split Data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(val_df)

        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Load Model (Reset for each fold)
        logger.info(f"Loading model from {MODEL_PATH}...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

        # Output Dir for this fold
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold+1}")

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5, # Default, similar to others
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            logging_dir=f"{fold_output_dir}/logs",
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            report_to="none"
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[LogCallback]
        )

        # Train
        logger.info(f"Starting training for Fold {fold+1}...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_train_time = time.time()
        trainer.train()
        end_train_time = time.time()
        
        fold_train_time = end_train_time - start_train_time
        train_times.append(fold_train_time)
        logger.info(f"Fold {fold+1} training time: {fold_train_time:.2f} seconds")

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            train_gpu_memory.append(peak_memory)
            logger.info(f"Fold {fold+1} Max GPU Memory (Train): {peak_memory:.2f} GB")
        else:
            train_gpu_memory.append(0.0)

        # Save Model for this fold
        logger.info(f"Saving model to {fold_output_dir}...")
        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)
        
        # Evaluate
        logger.info(f"Starting evaluation for Fold {fold+1}...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_eval_time = time.time()
        metrics = trainer.evaluate()
        end_eval_time = time.time()
        
        fold_eval_time = end_eval_time - start_eval_time
        eval_times.append(fold_eval_time)
        logger.info(f"Fold {fold+1} evaluation time: {fold_eval_time:.2f} seconds")
        logger.info(f"Fold {fold+1} metrics: {metrics}")

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            eval_gpu_memory.append(peak_memory)
            logger.info(f"Fold {fold+1} Max GPU Memory (Eval): {peak_memory:.2f} GB")
        else:
            eval_gpu_memory.append(0.0)
        
        fold_metrics.append(metrics)

    # Average Metrics
    logger.info(f"\n{'='*20} Cross-Validation Results {'='*20}")
    if fold_metrics:
        avg_accuracy = np.mean([m['eval_accuracy'] for m in fold_metrics])
        avg_f1 = np.mean([m['eval_f1'] for m in fold_metrics])
        avg_precision = np.mean([m['eval_precision'] for m in fold_metrics])
        avg_recall = np.mean([m['eval_recall'] for m in fold_metrics])
        
        avg_train_time = np.mean(train_times)
        avg_eval_time = np.mean(eval_times)
        
        avg_train_gpu = np.mean(train_gpu_memory)
        avg_eval_gpu = np.mean(eval_gpu_memory)

        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average F1 Score: {avg_f1:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")
        logger.info(f"Average Training Time: {avg_train_time:.2f} seconds")
        logger.info(f"Average Evaluation Time: {avg_eval_time:.2f} seconds")
        logger.info(f"Average Max GPU Memory (Train): {avg_train_gpu:.2f} GB")
        logger.info(f"Average Max GPU Memory (Eval): {avg_eval_gpu:.2f} GB")
    else:
        logger.info("No metrics collected.")
    logger.info("Done!")

if __name__ == "__main__":
    main()
