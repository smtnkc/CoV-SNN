import optuna
import subprocess
import re
import sys
import argparse

def objective(trial, target_neg_set=None):
    # 1. Suggest Hyperparameters
    neg_set = target_neg_set
    
    # Search Space
    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5])
    wd = trial.suggest_categorical("wd", [1e-5, 1e-4])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.5])
    projection_ratio = trial.suggest_categorical("projection_ratio", [0.2, 0.5])
    
    # Loss, Distance Metric, and Normalization
    loss_name = trial.suggest_categorical("loss_name", ["OnlineContrastiveLoss", "ContrastiveLoss"])
    distance_metric = trial.suggest_categorical("distance_metric", ["cosine", "euclidean"])

    # Cosine distance always requires normalization
    if distance_metric == "cosine":
        normalize_embeddings = True
    else: # Euclidean distance can be normalized or not
        normalize_embeddings = trial.suggest_categorical("normalize_embeddings", [True, False])

    # Conditional margin search based on distance metric
    if distance_metric == "cosine":
        margin = trial.suggest_float("margin", 0.05, 0.7)
    else:  # euclidean
        margin = trial.suggest_float("margin", 0.2, 1.2)

    pooling_mode = trial.suggest_categorical("pooling_mode", ["max", "mean"])

    # 2. Construct Command
    command = [
        "python", "-u", "train_test_zero_shot.py",  # Updated script name
        "--neg_set", str(neg_set),
        "--loss_name", str(loss_name),
        "--distance_metric", str(distance_metric),
        "--lr", str(lr),
        "--wd", str(wd),
        "--dropout", str(dropout),
        "--margin", str(margin),
        "--projection_ratio", str(projection_ratio),
        "--pooling_mode", str(pooling_mode),
        "--epochs", "10",
        "--batch_size", "32",
        "--early_stopping",
    ]
    
    if normalize_embeddings:
        command.append("--normalize_embeddings")

    print(f"Running trial {trial.number} with command: {' '.join(command)}")

    # 3. Run Subprocess
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        output_lines = []
        for line in process.stdout:
            print(line, end='') # Print to terminal
            output_lines.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Trial {trial.number} failed with return code {process.returncode}")
            return float('nan')
            
        output = "".join(output_lines)
            
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        return float('nan')

    # 4. Parse Output
    # Looking for: Best Test AUC: 0.1234...
    match = re.search(r"Best Test AUC: ([\d\.]+) at Epoch:", output)
    
    if match:
        best_test_auc = float(match.group(1))
        return best_test_auc
    else:
        print(f"Could not find 'Best Test AUC' in output for trial {trial.number}")
        return float('nan')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg_set", type=str, default=None, choices=["other", "delta"], help="Negative set to optimize for")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    study_name = f"contrastive_optimization_{args.neg_set}"
    storage_name = f"sqlite:///optuna_study_{args.neg_set}.db"
    
    # Create or load the study
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True
    )
    
    print(f"Start optimization... Study name: {study_name}")
    func = lambda trial: objective(trial, target_neg_set=args.neg_set)
    study.optimize(func, n_trials=args.n_trials) # Set a default, user can change or run repeatedly

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
