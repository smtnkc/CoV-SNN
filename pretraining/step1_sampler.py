import os
import pandas as pd

# Folder containing the CSV files
folder_path = "../data"

# List all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if (file.endswith(".csv") and not file.endswith("2k.csv"))]

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    print(f'Processing {file_path}...')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle cases where 'date' contains only the year and month
    year_month = df['date'].dt.strftime('%Y-%m')
    df.loc[df['date'].isnull(), 'date'] = year_month

    # Sort the DataFrame by the 'date' column
    df = df.sort_values(by='date')
    mask = df['sequence'].apply(len) <= 1278
    df_masked = df[mask].reset_index(drop=True)
    # Calculate the step size to sample 2000 items with the same distance
    step_size = len(df_masked) // 2000

    # Sample the DataFrame with the calculated step size
    sampled_df = df_masked.iloc[::step_size]
    sampled_df_tail = sampled_df.tail(2000)
    print(f"{len(df)} -> {len(df_masked)} --> {len(sampled_df)} -> {len(sampled_df_tail)}")

    # Export the sampled DataFrame to a new CSV file
    output_path = os.path.join(folder_path, f"{csv_file[:-4]}_2k.csv")
    sampled_df_tail.to_csv(output_path, index=False)

vocs = ["Alpha", "Beta", "Delta", "Gamma", "Omicron"]
seqs = []

for voc in vocs:
    seqs.extend(pd.read_csv(f"../data/unique_{voc}_2k.csv")["sequence"])
    print(len(seqs))
    
with open('../data/sequences_unique_10k.txt','w') as f:
    f.write('\n'.join(seqs))

file_path = '../data/sequences_unique_10k.txt'

max_length = 0

with open(file_path, 'r') as f:
    for line in f:
        line = line.rstrip('\n')       
        if len(line) > max_length:
            max_length = len(line)

print(f"The maximum length of a line in '{file_path}' is: {max_length}")
