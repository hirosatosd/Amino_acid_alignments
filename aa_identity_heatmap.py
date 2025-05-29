#!/usr/bin/env python3
"""
Amino Acid Percent Identity Heatmap Generator - Jupyter Notebook Version
This script calculates pairwise amino acid identity and creates a heatmap visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for Jupyter
%matplotlib inline
plt.style.use('default')
sns.set_palette("husl")

def calculate_pairwise_identity(seq1, seq2):
    """
    Calculate percent identity between two aligned sequences.
    Gaps are ignored in the calculation.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length (aligned)")
    
    matches = 0
    valid_positions = 0
    
    for a, b in zip(seq1, seq2):
        if a != '-' and b != '-':  # Skip gaps
            valid_positions += 1
            if a == b:
                matches += 1
    
    if valid_positions == 0:
        return 0.0
    
    return (matches / valid_positions) * 100

def read_sequences_from_fasta(fasta_file):
    """Read sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def read_sequences_from_alignment(alignment_file, format="fasta"):
    """Read sequences from alignment file."""
    sequences = {}
    alignment = AlignIO.read(alignment_file, format)
    for record in alignment:
        sequences[record.id] = str(record.seq)
    return sequences

def create_identity_matrix(sequences):
    """Create pairwise identity matrix from sequences."""
    seq_ids = list(sequences.keys())
    n = len(seq_ids)
    
    print(f"Calculating pairwise identities for {n} sequences...")
    
    # Initialize matrix
    identity_matrix = np.zeros((n, n))
    
    # Calculate pairwise identities
    for i in range(n):
        for j in range(n):
            if i == j:
                identity_matrix[i, j] = 100.0  # Self-identity is 100%
            else:
                identity = calculate_pairwise_identity(
                    sequences[seq_ids[i]], 
                    sequences[seq_ids[j]]
                )
                identity_matrix[i, j] = identity
    
    return pd.DataFrame(identity_matrix, index=seq_ids, columns=seq_ids)

def create_heatmap(identity_df, figsize=(12, 10), title="Amino Acid Identity Heatmap", 
                  save_file=None, show_values=True, colormap='RdYlBu_r'):
    """Create and display heatmap in Jupyter."""
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        identity_df, 
        annot=show_values, 
        fmt='.1f',
        cmap=colormap,
        center=50,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Percent Identity (%)'},
        annot_kws={'size': 8}
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Sequences', fontsize=12)
    plt.ylabel('Sequences', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_file}")
    
    plt.show()
    
    return plt.gcf()

def read_diamond_output(diamond_file):
    """
    Read DIAMOND output and create identity matrix.
    Expects tab-delimited format with columns: qseqid, sseqid, pident
    """
    df = pd.read_csv(diamond_file, sep='\t', header=None)
    
    # Assume standard DIAMOND output format
    if df.shape[1] >= 3:
        df.columns = ['qseqid', 'sseqid', 'pident'] + [f'col_{i}' for i in range(3, df.shape[1])]
    else:
        raise ValueError("DIAMOND file must have at least 3 columns (qseqid, sseqid, pident)")
    
    # Create pivot table for identity matrix
    identity_pivot = df.pivot_table(
        index='qseqid', 
        columns='sseqid', 
        values='pident', 
        fill_value=0
    )
    
    # Make symmetric matrix (add reverse matches if missing)
    all_seqs = set(identity_pivot.index) | set(identity_pivot.columns)
    identity_matrix = pd.DataFrame(0.0, index=all_seqs, columns=all_seqs)
    
    # Fill in the values
    for idx in identity_pivot.index:
        for col in identity_pivot.columns:
            val = identity_pivot.loc[idx, col]
            identity_matrix.loc[idx, col] = val
            identity_matrix.loc[col, idx] = val  # Make symmetric
    
    # Set diagonal to 100 (self-identity)
    for seq in all_seqs:
        identity_matrix.loc[seq, seq] = 100.0
    
    return identity_matrix

def analyze_sequences(input_file, input_format='fasta', title="AA Identity Heatmap", 
                     figsize=(12, 10), save_file=None, show_values=True, 
                     colormap='RdYlBu_r'):
    """
    Main function to analyze sequences and create heatmap.
    Perfect for Jupyter notebook usage.
    
    Parameters:
    -----------
    input_file : str
        Path to input file
    input_format : str
        Format of input file ('fasta', 'clustal', 'phylip', 'diamond')
    title : str
        Title for the heatmap
    figsize : tuple
        Figure size (width, height)
    save_file : str
        Optional file to save the plot
    show_values : bool
        Whether to show identity values on heatmap
    colormap : str
        Colormap for heatmap
    
    Returns:
    --------
    identity_df : pandas.DataFrame
        Matrix of pairwise identities
    """
    
    try:
        if input_format == 'diamond':
            print("Reading DIAMOND output...")
            identity_df = read_diamond_output(input_file)
        else:
            print(f"Reading sequences from {input_format} file...")
            if input_format == 'fasta':
                sequences = read_sequences_from_fasta(input_file)
            else:
                sequences = read_sequences_from_alignment(input_file, input_format)
            
            # Check if sequences are aligned
            seq_lengths = [len(seq) for seq in sequences.values()]
            if len(set(seq_lengths)) > 1:
                print("WARNING: Sequences have different lengths. Are they aligned?")
                print(f"Sequence lengths: {dict(zip(sequences.keys(), seq_lengths))}")
            
            identity_df = create_identity_matrix(sequences)
        
        print("Creating heatmap...")
        fig = create_heatmap(identity_df, figsize, title, save_file, show_values, colormap)
        
        # Print summary statistics
        upper_triangle = identity_df.values[np.triu_indices_from(identity_df.values, k=1)]
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Number of sequences: {len(identity_df)}")
        print(f"Mean identity: {upper_triangle.mean():.2f}%")
        print(f"Median identity: {np.median(upper_triangle):.2f}%")
        print(f"Min identity: {upper_triangle.min():.2f}%")
        print(f"Max identity: {upper_triangle.max():.2f}%")
        print(f"Standard deviation: {upper_triangle.std():.2f}%")
        
        # Show identity distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(upper_triangle, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Percent Identity (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pairwise Identities')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(upper_triangle)
        plt.ylabel('Percent Identity (%)')
        plt.title('Identity Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return identity_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Convenience function for quick analysis
def quick_heatmap(fasta_file, title=None):
    """Quick heatmap generation from FASTA file."""
    if title is None:
        title = f"Identity Heatmap - {Path(fasta_file).name}"
    
    return analyze_sequences(fasta_file, title=title)

# Example usage functions for Jupyter
def example_usage():
    """Print example usage for Jupyter notebooks."""
    print("EXAMPLE USAGE IN JUPYTER:")
    print("="*40)
    print()
    print("# Basic usage:")
    print("identity_matrix = analyze_sequences('alignment.fasta')")
    print()
    print("# With custom parameters:")
    print("identity_matrix = analyze_sequences(")
    print("    'alignment.fasta',")
    print("    title='My Protein Family',")
    print("    figsize=(15, 12),")
    print("    save_file='heatmap.png',")
    print("    colormap='viridis'")
    print(")")
    print()
    print("# Quick analysis:")
    print("identity_matrix = quick_heatmap('alignment.fasta')")
    print()
    print("# From DIAMOND output:")
    print("identity_matrix = analyze_sequences('diamond_output.txt', input_format='diamond')")
    print()
    print("# Access the data:")
    print("print(identity_matrix.head())")
    print("print(identity_matrix.loc['seq1', 'seq2'])  # Get specific identity")

# Display example usage when imported
print("AA Identity Heatmap Generator - Jupyter Ready!")
print("Call example_usage() to see usage examples.")
