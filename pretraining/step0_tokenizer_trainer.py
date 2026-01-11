from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.normalizers import BertNormalizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import RobertaProcessing
import pandas as pd
import time
from datetime import timedelta

vocs = ["Alpha", "Beta", "Delta", "Gamma", "Omicron"]
seqs = []

for voc in vocs:
    # Read the CSV file
    df = pd.read_csv(f"../data/unique_{voc}.csv")
    
    # Filter sequences longer than 1278
    filtered_seqs = df[df["sequence"].apply(len) <= 1278]["sequence"].tolist()
    
    # Extend the seqs list with the filtered sequences
    seqs.extend(filtered_seqs)
    
    print(f"{voc} -> {len(df)} -> {len(filtered_seqs)}")
    
# Write the sequences to a file
with open('../data/sequences.txt', 'w') as f:
    f.write('\n'.join(seqs))
    
tok = Tokenizer(BPE())

tok.pre_tokenizer = CharDelimiterSplit('\n')

trainer = BpeTrainer(
    vocab_size=10000,
    show_progress=True,
    initial_alphabet=["L","A","G","V","E","S","I","K","R","D",
                      "T","P","N","Q","F","Y","M","H","C","W",
                      "X","U","B","Z","O"],
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

start_time = time.time()

tok.train(files=["../data/sequences_unique_all.txt"], trainer=trainer)

tok.post_processor = RobertaProcessing(
    ("</s>", tok.token_to_id("</s>")),
    ("<s>", tok.token_to_id("<s>")),
)

elapsed_time = time.time() - start_time
formatted_time = str(timedelta(seconds=elapsed_time))
print(f"Elapsed time: {formatted_time}")

tok.save("../trained_tokenizer/tokenizer.json")
tok.model.save("../trained_tokenizer/")