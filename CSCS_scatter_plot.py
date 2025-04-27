import matplotlib.pyplot as plt
import pandas as pd
import sys

# Load the SentenceTransformer model
CHECKPOINT=sys.argv[1] # X for CoV-RoBERTa, 0 for CoV-SNN with best test accuracy, 4 for CoV-SNN with best zero-shot accuracy 
DISTANCE="L2" # L1 for Manhattan distance, L2 for Euclidean distance

# for GPT_TEMP in ["1.0", "1.5"]:
# Reading the data
df_eris_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_eris.csv")
df_new_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_new.csv")
df_gpt10_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_GPT_1.0.csv")
df_gpt15_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_GPT_1.5.csv")

# sort by semantic change descending and then by grammaticality ascending
df_eris_cscs = df_eris_cscs.sort_values(by=["semantic_change", "grammaticality"], ascending=[False, False])
df_new_cscs = df_new_cscs.sort_values(by=["grammaticality", "semantic_change"], ascending=[False, True])
df_gpt10_cscs = df_gpt10_cscs.sort_values(by=["grammaticality", "semantic_change"], ascending=[False, True])
df_gpt15_cscs = df_gpt15_cscs.sort_values(by=["semantic_change", "grammaticality"], ascending=[False, True])

# get top 100 to 200 samples
# df_eris_cscs = df_eris_cscs.iloc[10:100]
# df_new_cscs = df_new_cscs.iloc[10:100]
# df_gpt10_cscs = df_gpt10_cscs.iloc[10:100]
# df_gpt15_cscs = df_gpt15_cscs.iloc[10:100]

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ALPHA = 0.3
S = 10

# Font size settings
font_size = 20
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('figure', titlesize=font_size)

show_average = False

# Plotting Semantic Change vs Grammaticality
if show_average:
    # display only the average values
    axes[0].scatter(df_eris_cscs["log10(grammaticality)"].mean(), df_eris_cscs["log10(semantic_change)"].mean(), color='blue', alpha=1, s=100, label='Eris')
    axes[0].scatter(df_gpt10_cscs["log10(grammaticality)"].mean(), df_gpt10_cscs["log10(semantic_change)"].mean(), color='orange', alpha=1, s=100, label='GPT 1.0')
    axes[0].scatter(df_gpt15_cscs["log10(grammaticality)"].mean(), df_gpt15_cscs["log10(semantic_change)"].mean(), color='green', alpha=1, s=100, label='GPT 1.5')
    axes[0].scatter(df_new_cscs["log10(grammaticality)"].mean(), df_new_cscs["log10(semantic_change)"].mean(), color='red', alpha=1, s=100, label='New Omicron')
else:
    axes[0].scatter(df_eris_cscs["log10(grammaticality)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=ALPHA, s=S, label='Eris')
    axes[0].scatter(df_gpt10_cscs["log10(grammaticality)"], df_gpt10_cscs["log10(semantic_change)"], color='orange', alpha=ALPHA, s=S, label='GPT 1.0')
    axes[0].scatter(df_gpt15_cscs["log10(grammaticality)"], df_gpt15_cscs["log10(semantic_change)"], color='green', alpha=ALPHA, s=S, label='GPT 1.5')
    axes[0].scatter(df_new_cscs["log10(grammaticality)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=ALPHA, s=S, label='New Omicron')
axes[0].set_xlabel("Sequence Probability (log$_{10}$)", fontsize=font_size)
axes[0].set_ylabel("Semantic change (log$_{10}$)", fontsize=font_size)
axes[0].legend()
axes[0].set_title("Semantic Change vs Sequence Probability", fontsize=font_size)

# Plotting Semantic Change vs log10(1/Perplexity)
if show_average:
    axes[1].scatter(df_eris_cscs["log10(1/perplexity)"].mean(), df_eris_cscs["log10(semantic_change)"].mean(), color='blue', alpha=1, s=100, label='Eris')
    axes[1].scatter(df_gpt10_cscs["log10(1/perplexity)"].mean(), df_gpt10_cscs["log10(semantic_change)"].mean(), color='orange', alpha=1, s=100, label='GPT 1.0')
    axes[1].scatter(df_gpt15_cscs["log10(1/perplexity)"].mean(), df_gpt15_cscs["log10(semantic_change)"].mean(), color='green', alpha=1, s=100, label='GPT 1.5')
    axes[1].scatter(df_new_cscs["log10(1/perplexity)"].mean(), df_new_cscs["log10(semantic_change)"].mean(), color='red', alpha=1, s=100, label='New Omicron')
else:
    axes[1].scatter(df_eris_cscs["log10(1/perplexity)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=ALPHA, s=S, label='Eris')
    axes[1].scatter(df_gpt10_cscs["log10(1/perplexity)"], df_gpt10_cscs["log10(semantic_change)"], color='orange', alpha=ALPHA, s=S, label='GPT 1.0')
    axes[1].scatter(df_gpt15_cscs["log10(1/perplexity)"], df_gpt15_cscs["log10(semantic_change)"], color='green', alpha=ALPHA, s=S, label='GPT 1.5')
    axes[1].scatter(df_new_cscs["log10(1/perplexity)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=ALPHA, s=S, label='New Omicron')
axes[1].set_xlabel("Inverse Perplexity (log$_{10}$)", fontsize=font_size)
axes[1].set_ylabel("Semantic change (log$_{10}$)", fontsize=font_size)
axes[1].legend()
axes[1].set_title("Semantic Change vs Inverse Perplexity", fontsize=font_size)

# Explicitly setting the x-tick label size
for ax in axes:
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

# Overall title
if CHECKPOINT == "X":
    MODEL = "CoV-RoBERTa without Contrastive Learning"
elif CHECKPOINT == "4":
    for ax in axes:
        ax.legend(loc='center left')
    MODEL = "CoV-SNN Transformer with Best Zero-shot Acc"
elif CHECKPOINT == "0":
    MODEL = "CoV-SNN Transformer with Best Test Acc"
#plt.suptitle(f"Semantic Change Analysis ({MODEL})", fontsize=font_size)

# Layout and saving the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
AVG = "_avg" if show_average else ""
fig_file_name = f"figures/cscs_scatter_CP{CHECKPOINT}_{DISTANCE}{AVG}.pdf"
plt.savefig(fig_file_name)
# plt.show()