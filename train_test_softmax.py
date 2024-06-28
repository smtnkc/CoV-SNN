# %%
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
import numpy as np
import random
import argparse
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONSTANTS = {
    "VOC_NAMES": ["Alpha", "Beta", "Delta", "Gamma", "Omicron"],
    "LOSS_NAME": "SoftmaxLoss",
    "NEG_SET": "delta", # "other" or "delta"
    "POOLING_MODE": "max",
    "CONCAT": "CD", # "C" for concat-only, "CD" for concat+diff, or "CDM" for concat+diff+mult
    "NUM_LABELS": 2,
    "CONF_THRESHOLD": 0.0,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LR": 1e-3,
    "RELU": 0.1,
    "DROPOUT": 0.8,
    "MARGIN": None
}

parser = argparse.ArgumentParser(description='Process CONSTANTS.')
parser.add_argument('--loss_name', type=str, default="SoftmaxLoss", help='Loss function name')
parser.add_argument('--neg_set', type=str, default="delta", help='Negative set type')
parser.add_argument('--pooling_mode', type=str, default="max", help='Pooling mode')
parser.add_argument('--concat', type=str, default="CD", help='Concatenation type')
parser.add_argument('--num_labels', type=int, default=2, help='Number of labels')
parser.add_argument('--conf_threshold', type=float, default=0.0, help='Confidence threshold')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--relu', type=float, default=0.1, help='ReLU factor')
parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate')
parser.add_argument('--margin', type=float, default=None, help='Margin value')
parser.add_argument('--gstats', type=str, default="global_stats_softmax.csv", help='CSV file path')

args = parser.parse_args()
CONSTANTS["LOSS_NAME"] = args.loss_name
CONSTANTS["NEG_SET"] = args.neg_set
CONSTANTS["POOLING_MODE"] = args.pooling_mode
CONSTANTS["CONCAT"] = args.concat
CONSTANTS["NUM_LABELS"] = args.num_labels
CONSTANTS["CONF_THRESHOLD"] = args.conf_threshold
CONSTANTS["BATCH_SIZE"] = args.batch_size
CONSTANTS["EPOCHS"] = args.epochs
CONSTANTS["LR"] = args.lr
CONSTANTS["RELU"] = args.relu
CONSTANTS["DROPOUT"] = args.dropout
CONSTANTS["MARGIN"] = args.margin

if CONSTANTS["CONF_THRESHOLD"] is not None and CONSTANTS["CONF_THRESHOLD"] > 0 and (CONSTANTS["NUM_LABELS"] == 2 or CONSTANTS["NUM_LABELS"] == 5): 
    CONSTANTS["NUM_LABELS"] += 1 # add one more label for uncertain predictions

# %%
#word_embedding_model = models.Transformer(model_name_or_path="Rostlab/prot_bert", max_seq_length=1280)

encoder = models.Transformer(model_name_or_path="./mlm_checkpoints/CoV-RoBERTa_2048",
                                          max_seq_length=1280,
                                          tokenizer_name_or_path="tok/")

dim = encoder.get_word_embedding_dimension() # 768

pooler = models.Pooling(dim, pooling_mode = CONSTANTS["POOLING_MODE"])

modules = [encoder, pooler]

if CONSTANTS["RELU"] > 0:
    dense = models.Dense(in_features=dim, out_features=int(dim*CONSTANTS["RELU"]), activation_function=nn.ReLU())
    modules.append(dense)

if CONSTANTS["DROPOUT"] > 0:
    dropout = models.Dropout(CONSTANTS["DROPOUT"])
    modules.append(dropout)

model = SentenceTransformer(modules=modules)

# # Freeze initial transformer layers
# for param in model[0].auto_model.embeddings.parameters():
#     param.requires_grad = False
# for param in model[0].auto_model.encoder.layer[:6].parameters():
#     param.requires_grad = False

print(model)

# %% [markdown]
# # Construct Pairs for Training

# %%
omicron_sequences = pd.read_csv("data/unique_Omicron_2k.csv")["sequence"].tolist()
alpha_sequences = pd.read_csv("data/unique_Alpha_2k.csv")["sequence"].tolist()
beta_sequences = pd.read_csv("data/unique_Beta_2k.csv")["sequence"].tolist()
delta_sequences = pd.read_csv("data/unique_Delta_2k.csv")["sequence"].tolist()
gamma_sequences = pd.read_csv("data/unique_Gamma_2k.csv")["sequence"].tolist()

refs = []
seqs = []
labels = []
L = len(omicron_sequences)
if CONSTANTS["NEG_SET"] == "delta":
    for i in range(L):
        steps = [k*L//4 for k in range(1)] # range 1 means only one step, increase for more steps
        positives = [omicron_sequences[(i+step) % L] for step in steps]
        negatives = [delta_sequences[(i+step) % L] for step in steps]
        for neg, pos in zip(negatives, positives):
            refs.append(omicron_sequences[i])
            seqs.append(pos)
            labels.append(0) # 0 means omicron-omicron
            refs.append(omicron_sequences[i])
            seqs.append(neg)
            labels.append(1) # 1 means omicron-delta

elif CONSTANTS["NEG_SET"] == "other":
    NEGS = [delta_sequences, alpha_sequences, beta_sequences, gamma_sequences]
    for i in range(len(omicron_sequences)):
        for n_i, other_sequences in enumerate(NEGS):
            refs.append(omicron_sequences[i])
            seqs.append(omicron_sequences[(i+1) % len(omicron_sequences)])
            labels.append(0) # label = 0 means omicron-omicron
            refs.append(omicron_sequences[i])
            seqs.append(other_sequences[i])
            if CONSTANTS["NUM_LABELS"] == 2 or (CONSTANTS["CONF_THRESHOLD"] > 0 and CONSTANTS["NUM_LABELS"] == 3):
                labels.append(1) # label = 1 means omicron-other
            else:
                labels.append(n_i + 1) # label > 0 means omicron-other

df = pd.DataFrame({"reference": refs, "sequence": seqs, "label": labels})

examples = []
for i, row in df.iterrows():
    examples.append(InputExample(texts=[row["reference"], row["sequence"]], label=row["label"]))

print("Training set length:", len(examples))

# Split data into train, validation, and test sets
train_size = int(len(examples) * 0.8)
val_size = int(len(examples) * 0.1)
train_examples = examples[:train_size]
val_examples = examples[train_size:train_size + val_size]
test_examples = examples[train_size + val_size:]
# create dataloaders
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONSTANTS["BATCH_SIZE"])
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=CONSTANTS["BATCH_SIZE"])
test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=CONSTANTS["BATCH_SIZE"])

# %% [markdown]
# # Generate Pairs for Zero-shot Test

# %%
o = pd.read_csv("data/unique_Omicron_2k.csv")["sequence"].tolist()
e = pd.read_csv("data/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
n = pd.read_csv("data/unique_New_2k.csv")["sequence"].tolist()[:2000]

refs = []
seqs = []
labels = []

for i in range(len(omicron_sequences)):
        refs.append(o[i])
        seqs.append(e[i])
        labels.append(CONSTANTS["NUM_LABELS"]-1)
        refs.append(o[i])
        seqs.append(n[i])
        labels.append(0) # 0 means both are omicron
                
df = pd.DataFrame({"reference": refs, "sequence": seqs, "label": labels})

zero_test_examples = []
for i, row in df.iterrows():
    zero_test_examples.append(InputExample(texts=[row["reference"], row["sequence"]], label=row["label"]))

print("Zero-shot test set length: ", len(zero_test_examples))

# create dataloader for zero-shot test set
zero_test_dataloader = DataLoader(zero_test_examples, shuffle=False, batch_size=CONSTANTS["BATCH_SIZE"])

# %% [markdown]
# # Define Loss

# %%
softmax_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=CONSTANTS["NUM_LABELS"],
                                conf_threshold=CONSTANTS["CONF_THRESHOLD"],
                                concatenation_sent_rep = True if "C" in CONSTANTS["CONCAT"] else False,
                                concatenation_sent_difference = True if "D" in CONSTANTS["CONCAT"] else False,
                                concatenation_sent_multiplication = True if "M" in CONSTANTS["CONCAT"] else False)

print(softmax_loss)

# %% [markdown]
# # Construct Evaluators

# %%
evaluator = evaluation.LabelAccuracyEvaluator(val_dataloader, softmax_model=softmax_loss, name="Eval")
test_evaluator = evaluation.LabelAccuracyEvaluator(test_dataloader, softmax_model=softmax_loss, name="Test")
zero_test_evaluator = evaluation.LabelAccuracyEvaluator(zero_test_dataloader, softmax_model=softmax_loss, name="Zero")

# %% [markdown]
# # Prepare Folders

# %%
import os
import shutil

# Create output directory if needed
output_dir = f"./outputs/{CONSTANTS['LOSS_NAME']}_omicron_vs_{CONSTANTS['NEG_SET']}_" \
                f"P{CONSTANTS['POOLING_MODE']}_" \
                f"R{CONSTANTS['RELU']}_" \
                f"D{CONSTANTS['DROPOUT']}_" \
                f"E{CONSTANTS['EPOCHS']}_" \
                f"LR_{CONSTANTS['LR']}_" \
                f"B{CONSTANTS['BATCH_SIZE']}_" \
                f"T{CONSTANTS['CONF_THRESHOLD']}_" \
                f"{CONSTANTS['CONCAT']}"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed directory: {output_dir}")

checkpoint_dir = f"{output_dir}/checkpoints"
stats_dir = f"{output_dir}/stats"

for d in [checkpoint_dir, stats_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"Created directory: {d}")

# %% [markdown]
# # Run Training & Test

# %%
# print CONSTANTS
for k, v in CONSTANTS.items():
    print(f"{k}: {v}")

model.fit(train_objectives=[(train_dataloader, softmax_loss)],
          evaluator=evaluator,
          tester=test_evaluator,
          zero_shot_tester=zero_test_evaluator,
          epochs=CONSTANTS["EPOCHS"],
          optimizer_params= {'lr': CONSTANTS["LR"]}, # 1e-3 for CoV-RoBERTa, 1e-6 for ProtBERT
          weight_decay=0.1, # 0.1 for CoV-RoBERTa, 0.01 for ProtBERT
          #evaluation_steps=eval_iter,
          output_path=output_dir,
          save_best_model=True,
          #checkpoint_path=checkpoint_dir,
          #checkpoint_save_steps=len(train_dataloader), # save a checkpoint every epoch
          #checkpoint_save_total_limit=10000,
          show_progress_bar=False,
          loss_name=CONSTANTS["LOSS_NAME"])

# %% [markdown]
# # Display Stats

# %%
# read loss values from csv:
f_train_stats = os.path.join(stats_dir, 'Train.csv')
f_eval_stats = os.path.join(stats_dir, 'Eval.csv')
f_test_stats = os.path.join(stats_dir, 'Test.csv')
f_zero_stats = os.path.join(stats_dir, 'Zero.csv')

train_stats = pd.read_csv(f_train_stats)
eval_stats = pd.read_csv(f_eval_stats)
test_stats = pd.read_csv(f_test_stats)
zero_stats = pd.read_csv(f_zero_stats)

best_test_acc = test_stats["accuracy"].max()
best_zero_acc = zero_stats["accuracy"].max()

# create a dataframe with CONSTANTS and best accuracies
df = pd.DataFrame()
for k, v in CONSTANTS.items():
    if k not in ["VOC_NAMES"]:
        df[k] = [v] # if v is not None else ["N/A"]

df["MAX_TEST_ACC"] = best_test_acc
df["MAX_ZERO_ACC"] = best_zero_acc

print(df)

# save the dataframe to a csv file under stats_dir
df.to_csv(os.path.join(stats_dir, "summary.csv"), index=False)

# append row to global_stats.csv
global_stats_file = args.gstats
if not os.path.exists(global_stats_file) or os.path.getsize(global_stats_file) == 0:
    df.to_csv(global_stats_file, index=False)
else:
    global_stats = pd.read_csv(global_stats_file)
    global_stats = pd.concat([global_stats, df], ignore_index=True)
    global_stats.to_csv(global_stats_file, index=False)

# %%
# Plot the training and validation loss and accuracy
import matplotlib.pyplot as plt

# plot loss and accuracy figures side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(train_stats["epoch"], train_stats["loss"], label="Training")
axs[0].plot(eval_stats["epoch"], eval_stats["loss"], label="Validation")
axs[0].plot(test_stats["epoch"], test_stats["loss"], label="Test")
axs[0].plot(zero_stats["epoch"], zero_stats["loss"], label="Zero-shot")

axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[1].plot(train_stats["epoch"], train_stats["accuracy"], label="Training")
axs[1].plot(eval_stats["epoch"], eval_stats["accuracy"], label="Validation")
axs[1].plot(test_stats["epoch"], test_stats["accuracy"], label="Test")
axs[1].plot(zero_stats["epoch"], zero_stats["accuracy"], label="Zero-shot")

axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
plt.tight_layout()
#save as pdf
plt.savefig(os.path.join(stats_dir, "plot.pdf"))
#plt.show()
