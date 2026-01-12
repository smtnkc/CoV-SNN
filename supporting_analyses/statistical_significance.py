import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_source_from_name(name):
    """Extracts source (Eris, New, GPT) from sequence name or filename."""
    name_lower = name.lower()
    if 'eris' in name_lower:
        return 'Eris'
    elif 'new' in name_lower:
        return 'New'
    elif 'gpt' in name_lower:
        return 'GPT'
    return 'Other'

def load_bilstm_data(filepath):
    """Loads BiLSTM results and computes rank score."""
    print(f"Loading BiLSTM data from {filepath}...")
    try:
        df = pd.read_csv(filepath)

        # Ensure we have sc_per and gr_per.
        if 'sc_per' not in df.columns or 'gr_per' not in df.columns:
             print("  Warning: sc_per/gr_per not found. Using raw sc/gr.")
             if 'sc' in df.columns: df['sc_per'] = df['sc'] # Proxy
             if 'gr' in df.columns: df['gr_per'] = df['gr'] # Proxy
        
        df['rank_sc'] = df['sc_per'].rank(ascending=False)
        df['rank_gr'] = df['gr_per'].rank(ascending=False)
        df['score'] = df['rank_sc'] + df['rank_gr']

        # Source Label
        df['source'] = df['name'].apply(get_source_from_name)
        
        return df[['source', 'score']]
    except Exception as e:
        print(f"Error loading BiLSTM data: {e}")
        return None

def load_covsnn_data(base_path, subsets=['eris', 'new', 'GPT_1.0']):
    """Loads CoVSNN results for multiple subsets and combines them."""
    search_dir = os.path.dirname(base_path)
    dfs = []
    
    for subset in subsets:
        # Filename pattern: cscs_values_{subset}.csv
        # Note: subsets names in folder might differ case-wise.
        # Check file existence
        p = os.path.join(search_dir, f"cscs_values_{subset}.csv")
        if not os.path.exists(p):
             # Try lowercase/uppercase variants
             p_alt = os.path.join(search_dir, f"cscs_values_{subset.lower()}.csv")
             if os.path.exists(p_alt): p = p_alt
        
        if not os.path.exists(p):
            print(f"  Warning: CoVSNN file for {subset} not found at {p}")
            continue
            
        print(f"  Loading {subset} from {p}...")
        sub_df = pd.read_csv(p)
        sub_df['source'] = get_source_from_name(subset)
        dfs.append(sub_df)
        
    if not dfs:
        return None
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Ranking Logic from CoVSNN_rank.py:
    # rank_by_scip = rank(sc) + rank(ip)
    # where ip = 1/pp (Inverse Perplexity)
    
    if 'ip' not in df.columns:
        df['ip'] = 1 / df['perplexity']
        
    rank_sc = df['semantic_change'].rank(ascending=False)
    rank_ip = df['ip'].rank(ascending=False)
    
    df['score'] = rank_sc + rank_ip 
    
    return df[['source', 'score']]

def load_covfit_data(base_path_template, subsets=['Eris', 'New', 'Gpt_1.0']):
    """Loads CoVFit results."""
    # Base path template like: outputs/covfit_output_v4/{subset}/CoVFit_Predictions_fold_0.tsv
    
    dfs = []
    for subset in subsets:
        p = base_path_template.replace("{subset}", subset)
        if not os.path.exists(p):
             print(f"  Warning: CoVFit file for {subset} not found at {p}")
             continue
             
        print(f"  Loading {subset} from {p}...")
        sub_df = pd.read_csv(p, sep='\t')
        
        # Clean up
        if 'fitness_mean' in sub_df.columns:
            fitness = sub_df['fitness_mean']
        else:
            fitness = sub_df.iloc[:, 1]
            
        dms_cols = [c for c in sub_df.columns if c not in ['seq_name', 'fitness_mean', 'id', 'accession_id'] and not c.startswith('fitness_')]
        if not dms_cols:
             dms_values = sub_df.iloc[:, 2:1550]
        else:
            dms_values = sub_df[dms_cols]
            
        fitness = pd.to_numeric(fitness, errors='coerce')
        escape = dms_values.apply(pd.to_numeric, errors='coerce').mean(axis=1)
        
        # Rank Logic from CoVFit_analysis.py
        # Fitness Rank (Higher is better -> Rank 1)
        rank_fit = fitness.rank(ascending=False)
        # Escape Rank (Higher is better -> Rank 1)
        rank_esc = escape.rank(ascending=False)
        
        sub_df['score'] = (rank_fit + rank_esc) / 2 # Combined Rank
        sub_df['source'] = get_source_from_name(subset)
        
        dfs.append(sub_df[['source', 'score']])
        
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def calculate_precision_at_k(df, k, target_source='Eris'):
    """Calculates fraction of target_source in top K sorted by score (ascending)."""
    # Sort by score ascending (Rank 1 is best)
    top_k = df.sort_values(by='score', ascending=True).head(k)
    count = top_k['source'].value_counts().get(target_source, 0)
    return count / k

def bootstrap_precision_test(df_baseline, df_method, k=600, n_rounds=10000):
    """
    Bootstrap test for Precision@K.
    H0: Method Prec@K <= Baseline Prec@K
    H1: Method Prec@K > Baseline Prec@K
    """
    
    prec_diffs = []
    
    # Observed
    obs_prec_base = calculate_precision_at_k(df_baseline, k)
    obs_prec_method = calculate_precision_at_k(df_method, k)
    obs_diff = obs_prec_method - obs_prec_base
    
    n_base = len(df_baseline)
    n_method = len(df_method)
    
    print(f"  Observed Prec@{k}: Baseline={obs_prec_base:.4f}, Method={obs_prec_method:.4f}, Diff={obs_diff:.4f}")
    
    for _ in tqdm(range(n_rounds), desc=f"Bootstrapping Prec@{k}"):
        
        # Resample Baseline
        sample_base = df_baseline.sample(n=n_base, replace=True)
        p_base = calculate_precision_at_k(sample_base, k)
        
        # Resample Method
        sample_method = df_method.sample(n=n_method, replace=True)
        p_method = calculate_precision_at_k(sample_method, k)
        
        prec_diffs.append(p_method - p_base)
        
    prec_diffs = np.array(prec_diffs)
    
    # One-sided p-value: prob(diff <= 0)
    p_value = np.mean(prec_diffs <= 0)
    
    ci_lower = np.percentile(prec_diffs, 2.5)
    ci_upper = np.percentile(prec_diffs, 97.5)
    
    return obs_diff, p_value, ci_lower, ci_upper

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Paths
    bilstm_path = os.path.join(project_root, "BiLSTM", "outs", "cscs_scores_OR", "cscs_scores_avg_omic.csv")
    covsnn_path = os.path.join(project_root, "outputs", "CSCS", "CP4_L2", "cscs_values_eris.csv") # Used as base path
    
    # CoVFit Search
    covfit_base = None
    for v in ["covfit_output_v4", "covfit_output_v1", "escape_prediction/covfit_nov23"]:
        p = os.path.join(project_root, "outputs", v, "{subset}", "CoVFit_Predictions_fold_0.tsv")
        # Check if Eris exists
        if os.path.exists(p.replace("{subset}", "Eris")):
            covfit_base = p
            break
            
    print(f"CoVFit Pattern: {covfit_base}")
    
    # Load
    df_bilstm = load_bilstm_data(bilstm_path)
    df_covsnn = load_covsnn_data(covsnn_path, subsets=['eris', 'new', 'GPT_1.0'])
    df_covfit = load_covfit_data(covfit_base, subsets=['Eris', 'New', 'Gpt_1.0']) if covfit_base else None
    
    if df_bilstm is None: return
    
    K_VALUES = [300, 600, 900]
    
    for k in K_VALUES:
        print(f"\n=== Analysis for K={k} ===")
        
        if df_covsnn is not None:
            print("\nMap Check CoVSNN vs BiLSTM:")
            diff, p, ci_l, ci_u = bootstrap_precision_test(df_bilstm, df_covsnn, k=k)
            print(f"  Result: {'Significant' if p < 0.05 else 'Not Significant'} (p={p:.4e})")
            
        if df_covfit is not None:
            print("\nMap Check CoVFit vs BiLSTM:")
            diff, p, ci_l, ci_u = bootstrap_precision_test(df_bilstm, df_covfit, k=k)
            print(f"  Result: {'Significant' if p < 0.05 else 'Not Significant'} (p={p:.4e})")

if __name__ == "__main__":
    main()
