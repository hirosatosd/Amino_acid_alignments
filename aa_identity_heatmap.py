import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, pairwise2
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for Jupyter
%matplotlib inline

def simple_pairwise_align(seq1, seq2):
    """Simple pairwise alignment using Biopython."""
    alignments = pairwise2.align.globalxx(seq1, seq2)
    if alignments:
        return alignments[0][0], alignments[0][1]
    else:
        # Fallback: pad shorter sequence
        if len(seq1) < len(seq2):
            return seq1 + '-' * (len(seq2) - len(seq1)), seq2
        else:
            return seq1, seq2 + '-' * (len(seq1) - len(seq2))

def calculate_identity(seq1, seq2):
    """Calculate percent identity between two sequences."""
    original_seq1, original_seq2 = seq1, seq2
    
    # Align sequences if they're different lengths
    if len(seq1) != len(seq2):
        seq1, seq2 = simple_pairwise_align(seq1, seq2)
    
    matches = 0
    mismatches = 0
    gaps = 0
    
    for a, b in zip(seq1, seq2):
        if a == '-' or b == '-':  # Gap in either sequence
            gaps += 1
        elif a == b:  # Match
            matches += 1
        else:  # Mismatch
            mismatches += 1
    
    total_positions = len(seq1)
    valid_positions = matches + mismatches  # Non-gap positions
    
    # Calculate identity as matches / total alignment length (more stringent)
    # This penalizes gaps appropriately
    identity_total = (matches / total_positions) * 100
    
    # Also calculate traditional identity (matches / valid positions)
    identity_valid = (matches / valid_positions) * 100 if valid_positions > 0 else 0.0
    
    # Use the more conservative measure for very gappy alignments
    gap_fraction = gaps / total_positions
    
    if gap_fraction > 0.5:  # If more than 50% gaps, use total-based identity
        final_identity = identity_total
        method = "total-based"
    else:
        final_identity = identity_valid
        method = "valid-based"
    
    # Debug output for troubleshooting
    if len(original_seq1) != len(original_seq2):
        print(f"  Original lengths: {len(original_seq1)}, {len(original_seq2)}")
        print(f"  Aligned lengths: {len(seq1)}, {len(seq2)}")
        print(f"  Matches: {matches}, Mismatches: {mismatches}, Gaps: {gaps}")
        print(f"  Gap fraction: {gap_fraction:.2f}")
        print(f"  Identity (valid-based): {identity_valid:.1f}%")
        print(f"  Identity (total-based): {identity_total:.1f}%")
        print(f"  Final identity ({method}): {final_identity:.1f}%")
        print(f"  First 50 chars of alignment:")
        print(f"  Seq1: {seq1[:50]}")
        print(f"  Seq2: {seq2[:50]}")
        print()
    
    return final_identity

def make_heatmap(fasta_file, title="AA Identity Heatmap", figsize=(10, 8)):
    """
    Simple function to create identity heatmap from FASTA file.
    Handles both aligned and unaligned sequences.
    """
    
    # Read sequences
    print(f"Reading sequences from {fasta_file}...")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    
    seq_names = list(sequences.keys())
    n = len(seq_names)
    print(f"Found {n} sequences")
    
    # Calculate identity matrix
    print("Calculating pairwise identities...")
    identity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                identity_matrix[i, j] = 100.0
            else:
                identity = calculate_identity(sequences[seq_names[i]], sequences[seq_names[j]])
                identity_matrix[i, j] = identity
    
    # Create DataFrame
    identity_df = pd.DataFrame(identity_matrix, index=seq_names, columns=seq_names)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        identity_df, 
        annot=True, 
        fmt='.1f',
        cmap='RdYlBu_r',
        center=50,
        square=True,
        cbar_kws={'label': 'Identity (%)'}
    )
    
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print stats
    upper_tri = identity_df.values[np.triu_indices_from(identity_df.values, k=1)]
    
    # Check alignment lengths
    seq_lengths = [len(seq) for seq in sequences.values()]
    original_lengths = {name: len(sequences[name]) for name in seq_names}
    
    print(f"\nSequence Info:")
    print(f"Original lengths: {dict(list(original_lengths.items())[:5])}{'...' if len(original_lengths) > 5 else ''}")
    if len(set(seq_lengths)) == 1:
        print(f"All sequences aligned to length: {seq_lengths[0]}")
    else:
        print(f"Variable lengths: {set(seq_lengths)}")
    
    print(f"\nIdentity Stats:")
    print(f"Mean identity: {upper_tri.mean():.1f}%")
    print(f"Min identity: {upper_tri.min():.1f}%")
    print(f"Max identity: {upper_tri.max():.1f}%")
    
    return identity_df

# Usage example:
# identity_matrix = make_heatmap('your_sequences.fasta')
print("Simple AA Identity Heatmap ready!")
print("Usage: identity_matrix = make_heatmap('your_file.fasta')")
