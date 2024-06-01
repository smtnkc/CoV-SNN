#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_fasta_file accession_ids_file"
    exit 1
fi

input_fasta="$1"
accession_ids="$2"

# Check if input files exist
if [ ! -f "$input_fasta" ] || [ ! -f "$accession_ids" ]; then
    echo "Input file not found."
    exit 1
fi

# Create a temporary file to store the filtered sequences
temp_file=$(mktemp)

# Extract IDs from the FASTA file
awk '/^>/{print substr($1, 2)}' "$input_fasta" > "${input_fasta}.ids"

# Filter based on accession IDs
grep -Fwf "$accession_ids" "${input_fasta}.ids" > "${input_fasta}.filtered.ids"

# Filter the FASTA file using the filtered IDs
awk 'BEGIN {RS=">"; ORS=""} NR==FNR {ids[$1]; next} ($1 in ids) {print ">"$0}' "${input_fasta}.filtered.ids" "$input_fasta" > "$temp_file"

# Print the filtered sequences
cat "$temp_file"

# Clean up temporary files
rm -f "${input_fasta}.ids" "${input_fasta}.filtered.ids" "$temp_file"
