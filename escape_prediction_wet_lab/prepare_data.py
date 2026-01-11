import pandas as pd
from Bio import SeqIO
import os
import numpy as np


# Create output directory if it doesn't exist
os.makedirs('wet_lab_data', exist_ok=True)

# Read mutation files - no header
sig_mut_esc = pd.read_csv('wet_lab_data/sig_mut_esc.csv', header=None, names=['mutation']) # 2039 mutations
sig_mut_greaney = pd.read_csv('wet_lab_data/sig_mut_greaney.csv', header=None, names=['mutation']) # 181 mutations
sig_mut_baum = pd.read_csv('wet_lab_data/sig_mut_baum.csv', header=None, names=['mutation']) # 19 mutations

non_sig_mut_greaney = pd.read_csv('wet_lab_data/non_sig_mut_greaney.csv', header=None, names=['mutation']) # 2185 mutations
non_sig_mut_baum = pd.read_csv('wet_lab_data/non_sig_mut_baum.csv', header=None, names=['mutation']) # 24168 mutations (all possible non-escape mutations)
# All possible non-escape mutations = (20 possible AA - 1 actual AA) x 1273 positions - 19 significant mutations = 24168

# Filter sig_mut_esc to remove mutations present in sig_mut_greaney or sig_mut_baum
# Because we are using them for testing the model.
sig_mut_esc_filtered = sig_mut_esc[~sig_mut_esc['mutation'].isin(sig_mut_greaney['mutation'].tolist() + sig_mut_baum['mutation'].tolist())]
sig_mut_esc_filtered.to_csv('wet_lab_data/sig_mut_esc_filtered.csv', index=False, header=False) # 2010 mutations

# Filter non_sig_mut_baum to remove mutations present in sig_mut_greaney or sig_mut_esc
# Bist et. al. stated that some computationally generated non-escape mutantions are false negatives. 
# It means that they are listed as significant in the Greaney and ESC dataset.
sig_mutations = set(sig_mut_greaney['mutation'].tolist() + sig_mut_esc['mutation'].tolist()) # 2025 mutations
non_sig_mut_baum_filtered = non_sig_mut_baum[~non_sig_mut_baum['mutation'].isin(sig_mutations)]
non_sig_mut_baum_filtered.to_csv('wet_lab_data/non_sig_mut_baum_filtered.csv', index=False, header=False) # 22143 mutations

# Read wild type sequence
wild_type_seq = str(SeqIO.read('wet_lab_data/wild_type.fasta', 'fasta').seq)

def apply_mutations(mutations_df, output_file, is_greaney=False):

    print(f"Writing mutations to {output_file}")
    print(f"Mutation count: {len(mutations_df)}")

    """Apply mutations to wild type sequence and save results"""
    mutated_sequences = []
    valid_mutations = []
    
    for i, row in mutations_df.iterrows():
        mutation = row['mutation']
        orig_aa = mutation[0]
        pos = int(mutation[1:-1])
        new_aa = mutation[-1]
        
        # Convert to 0-based index if not greaney
        if not is_greaney:
            pos = pos - 1
        
        # Check if original amino acid matches
        if wild_type_seq[pos] != orig_aa:
            print(f"Warning {output_file} line {i}: Original amino acid mismatch for mutation {mutation}. Expected {wild_type_seq[pos]}, got {orig_aa}")
            continue
            
        mutated_seq = list(wild_type_seq)
        mutated_seq[pos] = new_aa
        mutated_sequences.append(''.join(mutated_seq))
        valid_mutations.append(mutation)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'mutation': valid_mutations,
        'sequence': mutated_sequences
    })
    output_df.to_csv(output_file, index=False, header=False)

    # Print name and length of output dataframe
    print(f"Sequence count: {len(output_df)}")

# Apply mutations and create sequence files
apply_mutations(sig_mut_esc, 'wet_lab_data/sig_seq_esc.csv', is_greaney=False)
apply_mutations(sig_mut_greaney, 'wet_lab_data/sig_seq_greaney.csv', is_greaney=True)
apply_mutations(sig_mut_baum, 'wet_lab_data/sig_seq_baum.csv', is_greaney=False)
apply_mutations(sig_mut_esc_filtered, 'wet_lab_data/sig_seq_esc_filtered.csv', is_greaney=False)
apply_mutations(non_sig_mut_baum, 'wet_lab_data/non_sig_seq_baum.csv', is_greaney=False)
apply_mutations(non_sig_mut_greaney, 'wet_lab_data/non_sig_seq_greaney.csv', is_greaney=True)
apply_mutations(non_sig_mut_baum_filtered, 'wet_lab_data/non_sig_seq_baum_filtered.csv', is_greaney=False)


# SPLIT THE DATA INTO TRAIN/VAL AND TEST SETS

# Read the sequence files
sig_seq_esc_filtered = pd.read_csv('wet_lab_data/sig_seq_esc_filtered.csv', header=None, names=['mutation', 'sequence'])
non_sig_seq_baum_filtered = pd.read_csv('wet_lab_data/non_sig_seq_baum_filtered.csv', header=None, names=['mutation', 'sequence'])

# Select the same number of non-escape sequences from non_sig_seq_baum_filtered as the number of escape sequences from sig_seq_esc_filtered
np.random.seed(42)
non_sig_train_val = non_sig_seq_baum_filtered.sample(n=len(sig_seq_esc_filtered))
non_sig_train_val.to_csv('wet_lab_data/non_sig_train_val.csv', index=False, header=False)
print(f"non_sig_train_val: {len(non_sig_train_val)}")

# sig_train_val will be the same as sig_seq_esc_filtered
sig_train_val = sig_seq_esc_filtered
sig_train_val.to_csv('wet_lab_data/sig_train_val.csv', index=False, header=False)
print(f"sig_train_val: {len(sig_train_val)}")


# EXTEND TRAINING DATA WITH GISAID SEQUENCES

sig_gisaid_seqs = []

# Parse the FASTA file and extract sequences
for record in SeqIO.parse('wet_lab_data/GISAID_SIG_SAMPLES.fa', 'fasta'):
    sig_gisaid_seqs.append(str(record.seq))

# remove sequences with X and len != 1273
sig_gisaid_seqs = [seq for seq in sig_gisaid_seqs if 'X' not in seq and len(seq) == 1273]
print(f"sig_gisaid_seqs: {len(sig_gisaid_seqs)}")

# check if any of the sequences are in the sig_train_val_seqs
sig_train_val_seqs = pd.read_csv('wet_lab_data/sig_train_val.csv', header=None, names=['mutation', 'sequence'])
print(f"sig_train_val_seqs: {len(sig_train_val_seqs)}")
sig_gisaid_seqs = [seq for seq in sig_gisaid_seqs if seq not in sig_train_val_seqs['sequence'].tolist()]
print(f"sig_gisaid_seqs after removing sig_train_val_seqs: {len(sig_gisaid_seqs)}")

# Add sig_gisaid_seqs to the sig_train_val_seqs dataframe
sig_train_val_seqs_extended = pd.concat(
    [sig_train_val_seqs, pd.DataFrame({'mutation': [''] * len(sig_gisaid_seqs), 'sequence': sig_gisaid_seqs})],
    ignore_index=True
)

# Shuffle the dataframe
sig_train_val_seqs_extended = sig_train_val_seqs_extended.sample(frac=1, random_state=42).reset_index(drop=True)
sig_train_val_seqs_extended.to_csv('wet_lab_data/sig_train_val_extended.csv', index=False, header=False)
print(f"sig_train_val_seqs after adding sig_gisaid_seqs: {len(sig_train_val_seqs_extended)}")

##########################

non_sig_gisaid_seqs = []

# Parse the FASTA file and extract sequences
for record in SeqIO.parse('wet_lab_data/GISAID_NON_SIG.fa', 'fasta'):
    non_sig_gisaid_seqs.append(str(record.seq))

# remove sequences with X and len != 1273
non_sig_gisaid_seqs = [seq for seq in non_sig_gisaid_seqs if 'X' not in seq and len(seq) == 1273]
print(f"non_sig_gisaid_seqs: {len(non_sig_gisaid_seqs)}")

# check if any of the sequences are in the non_sig_train_val_seqs
non_sig_train_val_seqs = pd.read_csv('wet_lab_data/non_sig_train_val.csv', header=None, names=['mutation', 'sequence'])
print(f"non_sig_train_val_seqs: {len(non_sig_train_val_seqs)}")
non_sig_gisaid_seqs = [seq for seq in non_sig_gisaid_seqs if seq not in non_sig_train_val_seqs['sequence'].tolist()]
print(f"non_sig_gisaid_seqs after removing non_sig_train_val_seqs: {len(non_sig_gisaid_seqs)}") 

# Add the sequences to the non_sig_train_val_seqs dataframe
non_sig_train_val_seqs_extended = pd.concat(
    [non_sig_train_val_seqs, pd.DataFrame({'mutation': [''] * len(non_sig_gisaid_seqs), 'sequence': non_sig_gisaid_seqs})],
    ignore_index=True
)

# Shuffle the dataframe
non_sig_train_val_seqs_extended = non_sig_train_val_seqs_extended.sample(frac=1, random_state=42).reset_index(drop=True)
non_sig_train_val_seqs_extended.to_csv('wet_lab_data/non_sig_train_val_extended.csv', index=False, header=False)
print(f"non_sig_train_val_seqs after adding non_sig_gisaid_seqs: {len(non_sig_train_val_seqs_extended)}")


# Update non_sig_mut_baum_filtered to remove mutations present in non_sig_train_val_seqs_extended
# Because we are using them for training the model
non_sig_train_val_seqs_extended = pd.read_csv('wet_lab_data/non_sig_train_val_extended.csv', header=None, names=['mutation', 'sequence'])
non_sig_mut_baum_filtered = non_sig_mut_baum_filtered[~non_sig_mut_baum_filtered['mutation'].isin(non_sig_train_val_seqs_extended['mutation'].tolist())]
non_sig_seq_baum_filtered = non_sig_seq_baum_filtered[~non_sig_seq_baum_filtered['mutation'].isin(non_sig_train_val_seqs_extended['mutation'].tolist())]
non_sig_mut_baum_filtered.to_csv('wet_lab_data/non_sig_mut_baum_filtered.csv', index=False, header=False)
non_sig_seq_baum_filtered.to_csv('wet_lab_data/non_sig_seq_baum_filtered.csv', index=False, header=False)

print(f"non_sig_seq_baum_filtered: {len(non_sig_seq_baum_filtered)}")


# check if non_sig_greaney_seqs and sig_train_val_seqs_extended have any common sequences
non_sig_greaney_seqs_df = pd.read_csv('wet_lab_data/non_sig_seq_greaney.csv', header=None, names=['mutation', 'sequence'])
non_sig_greaney_seqs_list = non_sig_greaney_seqs_df['sequence'].tolist()
print(f"non_sig_greaney_seqs: {len(set(non_sig_greaney_seqs_df['sequence'].tolist()))}")

non_sig_greaney_seqs_list_unique = list(set(non_sig_greaney_seqs_list))
print(f"non_sig_greaney_seqs unique: {len(non_sig_greaney_seqs_list_unique)}")

sig_train_val_extended_df = pd.read_csv('wet_lab_data/sig_train_val_extended.csv', header=None, names=['mutation', 'sequence'])
sig_train_val_extended_list = sig_train_val_extended_df['sequence'].tolist()
print(f"sig_train_val_extended: {len(sig_train_val_seqs)}")
sig_train_val_extended_list_unique = list(set(sig_train_val_extended_list))
print(f"sig_train_val_extended unique: {len(sig_train_val_extended_list_unique)}")

# Remove non_sig_greaney_seqs that are in sig_train_val_extended.
# These are false negatives. It means that they are listed as significant in the ESC dataset.
non_sig_greaney_seqs_list_filtered = [seq for seq in non_sig_greaney_seqs_list_unique if seq not in sig_train_val_extended_list_unique]
print(f"non_sig_greaney_seqs after removing sig_train_val_seqs: {len(non_sig_greaney_seqs_list_filtered)}")

# filter non_sig_greaney_seqs df using the remaining sequences
non_sig_greaney_seqs_df_filtered = non_sig_greaney_seqs_df[non_sig_greaney_seqs_df['sequence'].isin(non_sig_greaney_seqs_list_filtered)]
print(f"non_sig_greaney_seqs_df_filtered: {len(non_sig_greaney_seqs_df_filtered)}")

# Save the filtered sequences to a new file
non_sig_greaney_seqs_df_filtered.to_csv('wet_lab_data/non_sig_seq_greaney_filtered.csv', index=False, header=False)
