import pandas as pd
import numpy as np


df = pd.read_csv("SHORTER_THAN_1280/unique_10k.csv")

df["sequence"].str.len().max()
df["sequence"].str.len().min()
import csv

amino_acids = []
with open('SHORTER_THAN_1280/unique_10k.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        amino_acids.append(row[1])

amino_acids.remove("sequence")

print(len(amino_acids))


def convert_to_fasta(amino_acids):
    fasta_strings = []
    for i, sequence in enumerate(amino_acids):
        fasta_strings.append(f">sequence{i}\n{sequence}")
    return "\n".join(fasta_strings)

fasta_data = convert_to_fasta(amino_acids)

with open('input.fasta', 'w') as fastafile:
    fastafile.write(fasta_data)


import subprocess
from Bio.Align.Applications import ClustalwCommandline

cmd = ["apps/muscle", "-maxiters", "3", "-in", "input.fasta", "-out", "output.aln"]

# MUSCLE'ı çağırın
subprocess.run(cmd, check=True)

from Bio import AlignIO

# .aln uzantılı dosyayı oku
alignment = AlignIO.read("output.aln", "fasta")

# Hizalama sonuçlarını içeren bir .fasta dosyasına yaz
AlignIO.write(alignment, "alignment_output/alignment_output.fasta", "fasta")