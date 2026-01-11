
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Configuration ---
CONSTANTS = {
    "VOC_NAMES": ["Greaney", "Baum"],
    "MODEL_PATH": "./exp_outputs/Greaney/best_model", # Pointing to the best model
    "WT_FASTA": "exp_data/wild_type.fasta",
    "GREANEY_POS": "exp_data/non_sig_seq_greaney_filtered.csv",
    "GREANEY_NEG": "exp_data/sig_seq_greaney.csv",
    "BAUM_POS": "exp_data/non_sig_seq_baum_filtered.csv",
    "BAUM_NEG": "exp_data/sig_seq_baum.csv",
    "BATCH_SIZE": 32,
    "BOOTSTRAP_ROUNDS": 5000,
    "PERMUTATION_ROUNDS": 5000,
    "OUTPUT_PLOT": "exp_outputs/ROC_with_stats.pdf"
}

# --- Data Loading ---
from Bio import SeqIO

def load_data():
    # Load Wild Type
    wt_seq = str(SeqIO.read(CONSTANTS["WT_FASTA"], "fasta").seq)
    
    datasets = {}
    
    for name in CONSTANTS["VOC_NAMES"]:
        print(f"Loading data for {name}...")
        pos_file = CONSTANTS[f"{name.upper()}_POS"]
        neg_file = CONSTANTS[f"{name.upper()}_NEG"]
        
        # Load sequences (assuming no header, [mutation, sequence])
        # Based on previous notebooks, sometimes header=None, names=['mutation', 'sequence']
        try:
            pos_df = pd.read_csv(pos_file, header=None, names=['mutation', 'sequence'])
            neg_df = pd.read_csv(neg_file, header=None, names=['mutation', 'sequence'])
        except Exception as e:
            print(f"Error loading CSVs for {name}: {e}")
            continue
            
        pos_seqs = pos_df['sequence'].tolist()
        neg_seqs = neg_df['sequence'].tolist()
        
        datasets[name] = {
            "pos": pos_seqs,
            "neg": neg_seqs
        }
        print(f"  Positives: {len(pos_seqs)}, Negatives: {len(neg_seqs)}")
        
    return wt_seq, datasets

# --- bootstrapping functions ---
def stratified_bootstrap_auc(y_true, y_scores, n_rounds=1000):
    aucs = []
    
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    for _ in tqdm(range(n_rounds), desc="Bootstrapping AUC"):
        # Resample with replacement within classes
        pos_sample = np.random.choice(pos_indices, n_pos, replace=True)
        neg_sample = np.random.choice(neg_indices, n_neg, replace=True)
        
        sample_indices = np.concatenate([pos_sample, neg_sample])
        y_true_sample = y_true[sample_indices]
        y_scores_sample = y_scores[sample_indices]
        
        try:
            score = roc_auc_score(y_true_sample, y_scores_sample)
            aucs.append(score)
        except ValueError:
            continue # Should not happen if at least one of each class is present
            
    return np.array(aucs)

def permutation_test_p_value(y_true, y_scores, observed_auc, n_rounds=1000):
    perm_aucs = []
    y_true_perm = y_true.copy()
    
    for _ in tqdm(range(n_rounds), desc="Permutation Test"):
        np.random.shuffle(y_true_perm) # Shuffle labels
        try:
            score = roc_auc_score(y_true_perm, y_scores)
            perm_aucs.append(score)
        except ValueError:
            continue
            
    perm_aucs = np.array(perm_aucs)
    # One-sided p-value for H0: AUC = 0.5 vs H1: AUC > 0.5
    # Actually standard permutation test checks if observed statistic is extreme under null distribution
    # Null distribution is random association (AUC ~ 0.5)
    p_value = (np.sum(perm_aucs >= observed_auc) + 1) / (n_rounds + 1)
    return p_value, perm_aucs

# --- Main Execution ---

def main():
    # Load Data
    wt_seq, datasets = load_data()
    
    # Load Model
    print(f"Loading model from {CONSTANTS['MODEL_PATH']}...")
    # Ideally load directly if saved as SentenceTransformer
    # If not, might need to rebuild architecture and load weights. 
    # Try direct load first as `best_model` usually has config.json
    try:
        model = SentenceTransformer(CONSTANTS["MODEL_PATH"])
    except Exception as e:
        print(f"Could not load SentenceTransformer directly: {e}")
        print("Attempting to rebuild model structure (assuming CoV-RoBERTa + Pooling + Dense + Dropout)...")
        # Logic from step12 to rebuild if needed... 
        # For now assume it works, or fail fast. The directory listing showed 'best_model' folder.
        return

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Compute WT embedding (Anchor)
    print("Encoding Wild Type...")
    with torch.no_grad():
        wt_emb = model.encode([wt_seq], convert_to_tensor=True)
    
    results = {}
    
    for name, data in datasets.items():
        print(f"\nProcessing {name}...")
        pos_seqs = data['pos']
        neg_seqs = data['neg']
        all_test_seqs = pos_seqs + neg_seqs
        
        # Positives (Non-escape/Binds) => Label 1
        # Negatives (Escape/No-Bind) => Label 0 
        # Distance metric: Lower distance = Similar to WT = Non-escape (Label 1)
        # So Score = -Distance (Higher score = Label 1)
        
        y_true = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
        
        print(f"Encoding {len(all_test_seqs)} test sequences...")
        with torch.no_grad():
            test_embs = model.encode(all_test_seqs, batch_size=CONSTANTS["BATCH_SIZE"], convert_to_tensor=True, show_progress_bar=True)
            
        # Compute Euclidean Distances
        # wt_emb shape: [1, dim], test_embs shape: [N, dim]
        dists = torch.cdist(test_embs, wt_emb, p=2).cpu().numpy().flatten()
        
        # Function scores: closer to WT (small dist) => higher score (likely non-escape)
        y_scores = -dists
        
        # 1. Observed AUC
        obs_auc = roc_auc_score(y_true, y_scores)
        print(f"Observed AUC: {obs_auc:.4f}")
        
        # 2. Bootstrap 95% CI
        print(f"Running Bootstrap (B={CONSTANTS['BOOTSTRAP_ROUNDS']})...")
        boot_aucs = stratified_bootstrap_auc(y_true, y_scores, n_rounds=CONSTANTS["BOOTSTRAP_ROUNDS"])
        ci_lower = np.percentile(boot_aucs, 2.5)
        ci_upper = np.percentile(boot_aucs, 97.5)
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # 3. Permutation Test p-value
        print(f"Running Permutation Test (B={CONSTANTS['PERMUTATION_ROUNDS']})...")
        p_val, perm_aucs = permutation_test_p_value(y_true, y_scores, obs_auc, n_rounds=CONSTANTS['PERMUTATION_ROUNDS'])
        
        # Bonferroni Correction (2 datasets)
        p_val_adj = min(1.0, p_val * 2) 
        print(f"p-value: {p_val:.4e} (Adj: {p_val_adj:.4e})")
        
        # Store for plotting
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        results[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": obs_auc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_val": p_val,
            "p_val_adj": p_val_adj
        }

    # --- Plotting ---
    print("\nGenerating Plot...")
    plt.figure(figsize=(10, 8))
    
    colors = {"Greaney": "blue", "Baum": "darkorange"}
    
    for name, res in results.items():
        label = f"{name} (AUC = {res['auc']:.2f} [{res['ci_lower']:.2f}-{res['ci_upper']:.2f}])"
        # Add p-value to label if desired, or just in stats
        if res['p_val_adj'] < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {res['p_val_adj']:.3f}"
        
        label += f", {p_text}"
        
        plt.plot(res["fpr"], res["tpr"], color=colors.get(name, "black"), lw=2, label=label)
        
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curves with Statistical Validation", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.savefig(CONSTANTS["OUTPUT_PLOT"], bbox_inches="tight")
    print(f"Plot saved to {CONSTANTS['OUTPUT_PLOT']}")

if __name__ == "__main__":
    main()
