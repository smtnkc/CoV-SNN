import os
from tqdm import tqdm

input_file_path = "../data/sequences_unique_10k.txt"
masked_file_path = "../data/sequences_masked_12M.txt"
labels_file_path = "../data/labels_12M.txt"

max_length = 0

with open(input_file_path, 'r') as f:
    for line in f:
        line = line.rstrip('\n')       
        if len(line) > max_length:
            max_length = len(line)

print(f"The maximum length of a line in '{input_file_path}' is: {max_length}")

def aa_masker(input_string, index):
    # Create a list of characters to modify
    char_list = list(input_string)

    # Replace the character at the random index with "<mask>"
    label = char_list[index]
    char_list[index] = "<mask>"

    # Convert the list back to a string
    masked_string = ''.join(char_list)

    return masked_string, label

with open(input_file_path, "r") as input_file:
    sequences = input_file.read().splitlines()

# Initialize an empty list to store the masked sequences
masked_sequences = []
labels = []

pbar = tqdm(total=len(sequences), position=0)

# Create a tqdm progress bar for the entire process
for sequence in sequences:
    for index in range(len(sequence)):
        masked_sequence, label = aa_masker(sequence, index)
        masked_sequences.append(masked_sequence)
        labels.append(label)
    pbar.update(1)  # Update the progress bar for each processed sequence
pbar.close()

print(len(masked_sequences))

# Write the masked sequences to the output file
with open(masked_file_path, "w") as masked_file:
    for masked_sequence in masked_sequences:
        masked_file.write(masked_sequence + "\n")

# Write the masked sequences to the output file
with open(labels_file_path, "w") as labels_file:
    for label in labels:
        labels_file.write(label + "\n")

print(masked_sequences[10])
print()
print(f'<mask> = {labels[10]}')
