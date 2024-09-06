import pandas as pd
import numpy as np


# read wildtype sequence from txt file
with open('inputs/wt.txt') as f:
    wt = f.read()

# Load data
omic = pd.read_csv("inputs/unique_Omicron_2k.csv")["sequence"].tolist()[:2000]
eris = pd.read_csv("inputs/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
new = pd.read_csv("inputs/unique_New_2k.csv")["sequence"].tolist()[:2000]
gpt = pd.read_csv("inputs/unique_Gpt_2k.csv")["sequence"].tolist()[:2000]

# concat all sequence lists
all_sequences = {
    'wt': [wt],
    'omic': omic,
    'eris': eris,
    'new': new,
    'gpt': gpt
}

# convert all sequences to fasta format

def convert_to_fasta(all_sequences):
    fasta_strings = []
    for name, seq_list in all_sequences.items():
        for i, seq in enumerate(seq_list):
            fasta_strings.append(f">{name}_{i}\n{seq}")
    return "\n".join(fasta_strings)

fasta_data = convert_to_fasta(all_sequences)

with open('outputs/all_before_alignment.fasta', 'w') as fastafile:
    fastafile.write(fasta_data)




from Bio import AlignIO
# import subprocess
# from Bio.Align.Applications import ClustalwCommandline

# cmd = ["./muscle", "-maxiters", "3", "-in", "outputs/all_before_alignment.fasta", "-out", "outputs/all_after_alignment.aln"]

# # Call MUSCLE
# subprocess.run(cmd, check=True)

# # Read .aln file
# alignment = AlignIO.read("outputs/all_after_alignment.aln", "fasta")

# # Write aligned sequences to .fasta file
# AlignIO.write(alignment, "outputs/all_after_alignment.fasta", "fasta")

# Split aligned sequences into separate files for each variant
alignment = AlignIO.read("outputs/all_after_alignment.fasta", "fasta")
for record in alignment:
    variant = record.id.split("_")[0]
    with open(f"outputs/{variant}_after_alignment.fasta", "a") as f:
        f.write(f">{record.id}\n{record.seq}\n")
