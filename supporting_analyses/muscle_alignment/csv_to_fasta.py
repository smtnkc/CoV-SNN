import pandas as pd

# Load data
omic = pd.read_csv("inputs/unique_Omicron_2k.csv")["sequence"].tolist()[:2000]
eris = pd.read_csv("inputs/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
new = pd.read_csv("inputs/unique_New_2k.csv")["sequence"].tolist()[:2000]
gpt = pd.read_csv("inputs/unique_Gpt_2k.csv")["sequence"].tolist()[:2000]

sequences = {
    'omic': omic,
    'eris': eris,
    'new': new,
    'gpt': gpt
}


# convert all sequences to fasta format
def convert_to_fasta(sequences):
    fasta_strings = []
    for name, seq_list in sequences.items():
        for i, seq in enumerate(seq_list):
            fasta_strings.append(f">{name}_{i}\n{seq}")
    return "\n".join(fasta_strings)

# write separate fasta files for each variant
for name, seq_list in sequences.items():
    fasta_data = convert_to_fasta({name: seq_list})
    with open(f'outputs/{name}_before_alignment.fasta', 'w') as fastafile:
        fastafile.write(fasta_data)
