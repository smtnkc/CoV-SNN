import pandas as pd
import numpy as np
import csv
from Bio import AlignIO
import subprocess
from Bio.Align.Applications import ClustalwCommandline

variants = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Omicron', 'Eris']

for variant in variants:
    df = pd.read_csv(f"SHORTER_THAN_1280/unique_{variant}_2k.csv")

    seqs = []
    with open(f'SHORTER_THAN_1280/unique_{variant}_2k.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            seqs.append(row[1])

    seqs.remove("sequence")

    print(f"************* Number of {variant} sequences: {len(seqs)}")

    def convert_to_fasta(seqs):
        fasta_strings = []
        for i, sequence in enumerate(seqs):
            fasta_strings.append(f">sequence{i}\n{sequence}")
        return "\n".join(fasta_strings)

    fasta_data = convert_to_fasta(seqs)

    with open(f'input_{variant}.fasta', 'w') as fastafile:
        fastafile.write(fasta_data)


    cmd = ["/samet/muscle3.8", "-maxiters", "3", "-in", f"input_{variant}.fasta", "-out", f"output_{variant}.aln"]
    # /samet/muscle3.8 -maxiters 3 -in input.fasta -out output.aln

    # MUSCLE'ı çağırın
    subprocess.run(cmd, check=True)

    # .aln uzantılı dosyayı oku
    alignment = AlignIO.read(f"output_{variant}.aln", "fasta")

    # Hizalama sonuçlarını içeren bir .fasta dosyasına yaz
    AlignIO.write(alignment, f"alignment_output/{variant}.fasta", "fasta")