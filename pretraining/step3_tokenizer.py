import os
import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import time
from datetime import timedelta

tok = RobertaTokenizerFast.from_pretrained("../trained_tokenizer/")

# Read masked sequences and labels

masked_sequences = []
with open("../data/sequences_masked_12M.txt", 'r') as masked_file:
    for i, line in enumerate(masked_file):
        seq = line.strip()
        masked_sequences.append(seq)

labels_from_file = []
with open("../data/labels_12M.txt", "r") as labels_file:
    for i, line in enumerate(labels_file):
        lbl = line.strip()
        labels_from_file.append(lbl)

# Tokenize sequences and labels

data_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}

start_time = time.time()

pbar = tqdm(total=len(masked_sequences), position=0)
for i in range(len(masked_sequences)):
    temp_tok_seq = tok.encode_plus(masked_sequences[i])
    data_dict['input_ids'].append(temp_tok_seq['input_ids'])
    data_dict['attention_mask'].append(temp_tok_seq['attention_mask'])
    temp_tok_label = tok.encode(labels_from_file[i])[1]
    temp_tok_label_ext = [-100 if t != tok.mask_token_id else temp_tok_label for t in temp_tok_seq['input_ids']]
    data_dict['labels'].append(temp_tok_label_ext)
    pbar.update(1)  # Update the progress bar for each processed sequence
pbar.close()

elapsed_time = time.time() - start_time
formatted_time = str(timedelta(seconds=elapsed_time))
print(f"Elapsed time: {formatted_time}")

# Convert your data to PyTorch tensors
input_ids = [torch.tensor(sample) for sample in data_dict['input_ids']]
attention_mask = [torch.tensor(sample) for sample in data_dict['attention_mask']]
labels = [torch.tensor(sample) for sample in data_dict['labels']]

# Pad sequences to the same length
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tok.pad_token_id)
attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
labels = pad_sequence(labels, batch_first=True, padding_value=-100)

# Create a custom dataset
tds = TensorDataset(input_ids, attention_mask, labels)

torch.save(tds, "../data/tensor_dataset_12M.pth")
