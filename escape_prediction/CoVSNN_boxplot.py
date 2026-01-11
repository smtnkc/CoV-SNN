import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import fitz # pip install PyMuPDF

def get_results_df(CHECKPOINT, DISTANCE, DATASET):
    results_df = pd.read_csv(f"../outputs/CSCS/CP{CHECKPOINT}_{DISTANCE}/cscs_values_{DATASET}.csv")
    results_df.rename(columns={'semantic_change': 'sc'}, inplace=True)
    results_df.rename(columns={'grammaticality': 'sp'}, inplace=True)
    results_df.rename(columns={'perplexity': 'pp'}, inplace=True)
    results_df.rename(columns={'sentence': 'sequence'}, inplace=True)
    results_df["ip"] = 1 / results_df["pp"]
    results_df["gr"] = (results_df["sp"] + results_df["ip"]) / 2
    results_df["log10(sc)"] = np.log10(results_df["sc"])
    results_df["log10(sp)"] = np.log10(results_df["sp"])
    results_df["log10(ip)"] = np.log10(results_df["ip"])
    results_df['log10(gr)'] = np.log10(results_df['gr'])

    return results_df

def rank_results_df(results_df):
    # add rank_by_sc, rank_by_sp and rank_by_ip
    results_df["rank_by_sc"] = results_df["sc"].rank(ascending=False)
    results_df["rank_by_sp"] = results_df["sp"].rank(ascending=False)
    results_df["rank_by_ip"] = results_df["ip"].rank(ascending=False)
    results_df["rank_by_gr"] = results_df["gr"].rank(ascending=False)

    # make ranks integers
    results_df["rank_by_sc"] = results_df["rank_by_sc"].astype(int)
    results_df["rank_by_sp"] = results_df["rank_by_sp"].astype(int)
    results_df["rank_by_ip"] = results_df["rank_by_ip"].astype(int)
    results_df["rank_by_gr"] = results_df["rank_by_gr"].astype(int)

    # add rank_by_sc_sp, rank_by_sc_ip, and rank_by_sc_gr by adding the ranks of sc and sp/ip/gr
    results_df["rank_by_scsp"] = results_df["rank_by_sc"] + results_df["rank_by_sp"]
    results_df["rank_by_scip"] = results_df["rank_by_sc"] + results_df["rank_by_ip"]
    results_df["rank_by_scgr"] = results_df["rank_by_sc"] + results_df["rank_by_gr"]

    # By default sort by rank_by_sc_gr
    results_df = results_df.sort_values(by="rank_by_scgr")
    return results_df

def remove_outliers(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def draw(CHECKPOINT, DISTANCE, GPT_TEMP):
    # Overall title
    if CHECKPOINT == "X":
        TITLE = "CoV-RoBERTa without Contrastive Learning"
    elif CHECKPOINT == "4":
        TITLE = "CSCS Ranking Distributions at Epoch 5"
    elif CHECKPOINT == "0":
        TITLE = "CSCS Ranking Distributions at Epoch 1"

    # Reading the data
    df_eris_cscs = get_results_df(CHECKPOINT, DISTANCE, "eris")
    df_new_cscs = get_results_df(CHECKPOINT, DISTANCE, "new")
    dfs_gpt_cscs = []
    for temp in GPT_TEMP:
        dfs_gpt_cscs.append(get_results_df(CHECKPOINT, DISTANCE, f"GPT_{temp}"))

    df_eris_cscs = remove_outliers(df_eris_cscs, 'sc')
    df_new_cscs = remove_outliers(df_new_cscs, 'sc')
    for i in range(len(GPT_TEMP)):
        dfs_gpt_cscs[i] = remove_outliers(dfs_gpt_cscs[i], 'sc')

    # Combine the three DataFrames into one for easier plotting
    df_combined = pd.concat([
        df_eris_cscs.assign(source='Eris'),
        df_new_cscs.assign(source='New')
    ])
    for i, temp in enumerate(GPT_TEMP):
        df_combined = pd.concat([
            df_combined,
            dfs_gpt_cscs[i].assign(source=f'GPT$_{{{temp}}}$')
        ])

    df_combined = rank_results_df(df_combined)

    # Convert rank values to percentages if needed
    df_combined['rank_by_scsp'] = (df_combined['rank_by_scsp'] / df_combined['rank_by_scsp'].max()) * 100
    df_combined['rank_by_scip'] = (df_combined['rank_by_scip'] / df_combined['rank_by_scip'].max()) * 100

    # Plotting the boxplots using seaborn
    sns.set_theme(style="whitegrid", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(
        data=df_combined,
        x='source',
        y='rank_by_scsp',
        width=0.75,
        ax=axes[0],
        showfliers=True
    )
    sns.set_style("white")
    axes[0].set_title(r"R$'$ = rank($\Delta_Z$) + rank($\lambda_T$)")
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Ranked in Top (%)')
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[0].set_ylim(-5, 105)  # Set consistent spacing for 0% and 100%

    sns.boxplot(
        data=df_combined,
        x='source',
        y='rank_by_scip',
        ax=axes[1],
        width=0.75,
        showfliers=True
    )
    sns.set_style("white")
    axes[1].set_title(r"R$'$ = rank($\Delta_Z$) + rank($\lambda_A$)")
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Ranked in Top (%)')
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].set_ylim(-5, 105)  # Set consistent spacing for 0% and 100%

    # hide gridlines
    axes[0].grid(False)
    axes[1].grid(False)

    fig.suptitle(TITLE, fontsize=16, y=1.05)
    # set size of figure
    fig.set_size_inches(10, 4.5)
    plt.tight_layout()
    
    fig_file_name = f"../outputs/CSCS/cscs_box_CP{CHECKPOINT}_{DISTANCE}.pdf"
    plt.savefig(fig_file_name, tboxinches='tight', bbox_inches='tight', dpi=300)
    #plt.show()
    return fig, fig_file_name


def merge_figures(fig_path1, fig_path2):
    # Load the two PDF images
    doc1 = fitz.open(fig_path1)
    doc2 = fitz.open(fig_path2)

    # Extract the first (and only) page from each
    page1 = doc1[0]
    page2 = doc2[0]

    # Define spacing between images (in points)
    gap = 15

    # Compute combined dimensions
    w = max(page1.rect.width, page2.rect.width)
    h = page1.rect.height + page2.rect.height + gap

    # Create a new blank page
    merged_doc = fitz.open()
    new_page = merged_doc.new_page(width=w, height=h)

    # Track vertical position from top (y = 0)
    y = 0

    # Insert first image (top)
    new_page.show_pdf_page(
        fitz.Rect(0, y, w, y + page1.rect.height),
        doc1,
        0
    )
    y += page1.rect.height + gap

    # Insert second image (bottom)
    new_page.show_pdf_page(
        fitz.Rect(0, y, w, y + page2.rect.height),
        doc2,
        0
    )

    merged_doc.save("../outputs/CSCS/fig_merged_box.pdf")

if __name__ == "__main__":
    fig1, fig_file_path1 = draw(CHECKPOINT="0", DISTANCE="L2", GPT_TEMP=["1.0", "1.5"])
    fig2, fig_file_path2 = draw(CHECKPOINT="4", DISTANCE="L2", GPT_TEMP=["1.0", "1.5"])
    merge_figures(fig_file_path1, fig_file_path2)



