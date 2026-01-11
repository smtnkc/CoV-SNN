import os
import subprocess
import pandas as pd
import argparse
import sys
from pathlib import Path

# --- Configuration ---
# Instructions for downloading CovFit CLI:
# 1. Go to https://github.com/TheSatoLab/CoVFit/blob/main/CoVFit_CLI/ReadMe.md
# 2. Download the appropriate executable for your OS.
# 3. Place it in the same directory as this script or update COVFIT_CLI_PATH.
# 4. Make it executable: chmod +x covfit_cli


DATA_DIR = "../data"

# Mapping of dataset names to their filenames in the data directory
DATASETS = {
    "Eris": "unique_Eris_2k.csv",
    "New": "unique_New_2k.csv",
    "Gpt_1.0": "unique_Gpt_1.0_2k.csv",
}

def convert_csv_to_fasta(csv_path, fasta_path, limit=None):
    """Converts a CSV with accession_id and sequence columns to FASTA format."""
    try:
        df = pd.read_csv(csv_path)
        
        if limit and len(df) > limit:
            print(f"Limiting {csv_path} to first {limit} sequences.")
            df = df.head(limit)
            
        with open(fasta_path, 'w') as f:
            for _, row in df.iterrows():
                # Assuming columns 'accession_id' and 'sequence' based on previous inspection
                header = row.get('accession_id', row.get('id', 'unknown'))
                seq = row.get('sequence', row.get('seq', ''))
                if seq:
                    f.write(f">{header}\n{seq}\n")
        print(f"Converted {csv_path} to {fasta_path} ({len(df)} sequences)")
        return True
    except Exception as e:
        print(f"Error converting {csv_path}: {e}")
        return False

def run_covfit(input_fasta, output_dir, use_gpu=True, batch_size=4, dry_run=False, model="noDMS", fold=0):
    """Runs the CovFit CLI command with DMS scores enabled."""
    
    # Ensure output directory exists (CovFit CLI might expect it or create it, safe to create parent)
    os.makedirs(output_dir, exist_ok=True)
    
    if model == "noDMS":
        COVFIT_PY_PATH = "../../CoVFit_CLI_noDMS/_internal/files/covfit_cli_patched.py"
        cmd = [
        "python3",
        COVFIT_PY_PATH,
        "--input", input_fasta,
        "--outdir", output_dir,
        "--batch", str(batch_size),
        "--fold", str(fold)
    ]
    elif model == "Nov23":
        COVFIT_CLI_PATH = "../../CoVFit_CLI_v1/covfit_cli"
        cmd = [
            COVFIT_CLI_PATH,
            "--input", input_fasta,
            "--outdir", output_dir,
            "--batch", str(batch_size),
            "--fold", str(fold),
            "--dms"  # Enable DMS outputs
        ]
    
    if use_gpu:
        cmd.append("--gpu")
        
    print(f"Running command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] Command skipped.")
        return True
        
    check_path = COVFIT_CLI_PATH if model == "Nov23" else COVFIT_PY_PATH

    if not os.path.exists(check_path):
        print(f"Error: CovFit CLI executable not found at {check_path}")
        print("Please download it from https://github.com/TheSatoLab/CoVFit/tree/main/CoVFit_CLI")
        return False

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"CovFit execution failed: {e}")
        return False

def get_results_dataframe(dataset_name, output_dir, fold=0):
    """Reads the output TSV from CovFit and returns a DataFrame with Fitness and Escape scores."""
    result_file = os.path.join(output_dir, f"CoVFit_Predictions_fold_{fold}.tsv")
    
    # Handle potential capitalization diffs based on past runs
    if not os.path.exists(result_file):
        result_file = os.path.join(output_dir, f"CoVFit_Predictions_Fold_{fold}.tsv")
        
    if not os.path.exists(result_file):
        print(f"Warning: Result file {result_file} not found for {dataset_name}")
        return None
        
    try:
        # Load the TSV
        df = pd.read_csv(result_file, sep='\t')
        
        # 0. Identify Fitness Column
        if 'fitness_mean' in df.columns:
            fitness_values = df['fitness_mean']
        else:
            # Fallback based on column index if names are missing (unlikely given sample)
            # Sample col 1 is fitness_mean (0 is seq_name)
            fitness_values = df.iloc[:, 1]
            
        # 1. Identify DMS Columns
        # Filter columns that are NOT: 'seq_name', 'fitness_mean', or start with 'fitness_'
        dms_cols = [c for c in df.columns if c not in ['seq_name', 'fitness_mean', 'id', 'accession_id'] and not c.startswith('fitness_')]
        
        if not dms_cols:
            print(f"Warning: No DMS columns found for {dataset_name}. Using columns 2-1550 as fallback.")
            dms_values = df.iloc[:, 2:1550]
        else:
            dms_values = df[dms_cols]
            
        print(f"DEBUG: {dataset_name} - Found {len(dms_cols)} DMS columns. First: {dms_cols[0]}, Last: {dms_cols[-1]}")
        
        # Calculate scores
        # Ensure numeric
        fitness_values = pd.to_numeric(fitness_values, errors='coerce')
        dms_values = dms_values.apply(pd.to_numeric, errors='coerce')
        
        # Escape Score = Mean of DMS columns
        escape_scores = dms_values.mean(axis=1)
        
        clean_df = pd.DataFrame({
            'Fitness': fitness_values,
            'Escape': escape_scores,
            'Dataset': dataset_name
        })
        
        # Drop rows where Fitness is NaN (valid sequences only)
        clean_df = clean_df.dropna(subset=['Fitness'])
        
        return clean_df
    except Exception as e:
        print(f"Error analyzing results for {dataset_name}: {e}")
        return None

def analyze_combined_rankings(results_df):
    """Calculates combined ranking and distribution."""
    print("\n--- Combined Ranking Analysis (Fitness + Escape) ---")
    
    results_df = results_df.dropna(subset=['Fitness', 'Escape'])
    
    # 1. Rank by Fitness (Higher is better)
    # rank(ascending=False) means highest value gets rank 1
    results_df['Fitness_Rank'] = results_df['Fitness'].rank(ascending=False)
    
    # Correlation Check
    correlation = results_df['Fitness'].corr(results_df['Escape'])
    print(f"Correlation between Fitness and Escape Score: {correlation:.4f}")
    if correlation > 0.9:
        print("  ! High correlation detected. Escape score adds little new information.")

    # 2. Rank by Escape (Higher mean DMS score is better - assuming higher DMS prediction = higher escape/fitness gain?)
    # "mutation effect information related to the ability of the virus to evade humoral immunity"
    # "in silico DMS... infer fitness gain"
    # Usually higher score = higher escape/fitness.
    results_df['Escape_Rank'] = results_df['Escape'].rank(ascending=False)
    
    # 3. Combined Rank
    results_df['Combined_Rank'] = (results_df['Fitness_Rank'] + results_df['Escape_Rank']) / 2
    
    # Sort by Combined Rank (ascending, lower is better)
    sorted_df = results_df.sort_values(by='Combined_Rank', ascending=True).reset_index(drop=True)
    
    total_count = len(sorted_df)
    print(f"Total sequences analyzed: {total_count}")
    
    thresholds = [0.05, 0.10, 0.15, 0.20]
    
    summary_data = []

    for k in thresholds:
        top_k_count = int(total_count * k)
        if top_k_count == 0:
            continue
            
        top_k_df = sorted_df.head(top_k_count)
        
        print(f"\nTop {k*100:.0f}% ({top_k_count} seqs):")
        
        counts = top_k_df['Dataset'].value_counts()
        for dataset_name, count in counts.items():
            percent = (count / top_k_count) * 100
            print(f"  {dataset_name}: {percent:.2f}% ({count} seqs)")
            
            summary_data.append({
                "Top_Threshold": f"{k*100:.0f}%",
                "Dataset": dataset_name,
                "Count": count,
                "Percentage": percent
            })
        
    return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description="Run CovFit Combined Analysis.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare data but do not run CovFit.")
    parser.add_argument("--model", type=str, default="Nov23", choices=["noDMS", "Nov23"], help="CoVFit model to use.")
    parser.add_argument("--fold", type=int, default=0, help="Fold number.")

    args = parser.parse_args()

    all_results = []
    
    if args.model == "noDMS":
        OUTPUT_DIR = "../outputs/escape_prediction/covfit_nodms"
    elif args.model == "Nov23":
        OUTPUT_DIR = "../outputs/escape_prediction/covfit_nov23"

    for name, filename in DATASETS.items():
        print(f"\nProcessing {name}...")
        csv_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(csv_path):
            print(f"  Warning: Data file {csv_path} not found. Skipping.")
            continue
            
        dataset_out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        fasta_path = os.path.join(dataset_out_dir, f"{name}.fasta")
        
        # 1. Convert
        if not convert_csv_to_fasta(csv_path, fasta_path, limit=2000):
            continue
            
        # 2. Run CovFit
        success = run_covfit(fasta_path, dataset_out_dir, use_gpu=args.gpu, batch_size=args.batch_size,
            dry_run=args.dry_run, model=args.model, fold=args.fold)
        
        if success and not args.dry_run:
            # 3. Analyze
            df = get_results_dataframe(name, dataset_out_dir, fold=args.fold)
            if df is not None:
                all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Analyze Rankings
        ranking_summary = analyze_combined_rankings(combined_df)
        
        # Save Analysis
        analysis_path = os.path.join(OUTPUT_DIR, f"combined_ranking_analysis_fold_{args.fold}.csv")
        ranking_summary.to_csv(analysis_path, index=False)
        print(f"Ranking analysis saved to {analysis_path}")

if __name__ == "__main__":
    main()
