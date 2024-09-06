import matplotlib.pyplot as plt
import pandas as pd
import sys

# Load the SentenceTransformer model
CHECKPOINT=sys.argv[1] # X for CoV-RoBERTa, 0 for CoV-SNN with best test accuracy, 4 for CoV-SNN with best zero-shot accuracy 
DISTANCE="L2" # L1 for Manhattan distance, L2 for Euclidean distance

for GPT_TEMP in ["1.0", "1.1", "1.2"]:
    # Reading the data
    df_eris_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_eris.csv")
    df_new_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_new.csv")
    df_gpt_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_GPT_{GPT_TEMP}.csv")

    # Creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ALPHA = 0.3
    S = 10

    # Font size settings
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('figure', titlesize=font_size)

    # Plotting Semantic Change vs Grammaticality
    axes[0].scatter(df_eris_cscs["log10(grammaticality)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=ALPHA, s=S, label='Eris')
    axes[0].scatter(df_gpt_cscs["log10(grammaticality)"], df_gpt_cscs["log10(semantic_change)"], color='orange', alpha=ALPHA, s=S, label=f'GPT {GPT_TEMP}')
    axes[0].scatter(df_new_cscs["log10(grammaticality)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=ALPHA, s=S, label='New Omicron')
    axes[0].set_xlabel("Sequence Probability (log$_{10}$)", fontsize=16)
    axes[0].set_ylabel("Semantic change (log$_{10}$)", fontsize=16)
    axes[0].legend()
    axes[0].set_title("Semantic Change vs Sequence Probability", fontsize=16)

    # Plotting Semantic Change vs log10(1/Perplexity)
    axes[1].scatter(df_eris_cscs["log10(1/perplexity)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=ALPHA, s=S, label='Eris')
    axes[1].scatter(df_gpt_cscs["log10(1/perplexity)"], df_gpt_cscs["log10(semantic_change)"], color='orange', alpha=ALPHA, s=S, label=f'GPT {GPT_TEMP}')
    axes[1].scatter(df_new_cscs["log10(1/perplexity)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=ALPHA, s=S, label='New Omicron')
    axes[1].set_xlabel("Inverse Perplexity (log$_{10}$)", fontsize=16)
    axes[1].set_ylabel("Semantic change (log$_{10}$)", fontsize=16)
    axes[1].legend()
    axes[1].set_title("Semantic Change vs Inverse Perplexity", fontsize=16)


    # Overall title
    if CHECKPOINT == "X":
        MODEL = "CoV-RoBERTa without Contrastive Learning"
    elif CHECKPOINT == "4":
        MODEL = "CoV-SNN Transformer with Best Zero-shot Acc"
    elif CHECKPOINT == "0":
        MODEL = "CoV-SNN Transformer with Best Test Acc"
    #plt.suptitle(f"Semantic Change Analysis ({MODEL})", fontsize=font_size)

    # Layout and saving the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_file_name = f"figures/cscs_scatter_CP{CHECKPOINT}_{DISTANCE}_{GPT_TEMP}.pdf"
    plt.savefig(fig_file_name)
# plt.show()