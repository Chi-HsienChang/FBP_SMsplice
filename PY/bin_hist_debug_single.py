#!/usr/bin/env python3

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Get dataset and top_k value from command-line arguments
# Usage: python3 script.py t 100
dataset = sys.argv[1]  # e.g., "t", "z", "m", or "h"
top_k = int(sys.argv[2])

###############################################################################
# 1) Function to parse a splice site file
###############################################################################

def parse_splice_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Patterns for annotated splice sites
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")

    def parse_annotated_sites(regex):
        match = regex.search(text)
        if not match:
            return set()
        inside = match.group(1).strip()
        if not inside:
            return set()
        sites = re.split(r"[\s,]+", inside.strip())
        return set(map(int, sites))

    annotated_5prime = parse_annotated_sites(pattern_5ss)
    annotated_3prime = parse_annotated_sites(pattern_3ss)

    # Patterns for sorted predictions (needed to calculate fraction correct)
    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)",
        re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(pattern):
        match_block = pattern.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        preds = []
        for m in pattern_line.finditer(block):
            pos = int(m.group(1))
            prob = float(m.group(2))
            preds.append((pos, prob))
        return preds

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    rows = []
    for (pos, prob) in fiveprime_preds:
        is_correct = (pos in annotated_5prime)
        rows.append((pos, prob, "5prime", is_correct))
    for (pos, prob) in threeprime_preds:
        is_correct = (pos in annotated_3prime)
        rows.append((pos, prob, "3prime", is_correct))
    
    df = pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct"])
    return df

###############################################################################
# 2) Helper function: Plot histogram for a given category
#    category: "5SS", "3SS", or "ALL" (for combined data)
#    Bins: a ≤ p < b with last bin: p ≥ 0.9
###############################################################################

def plot_histogram(category, dataset, top_k, df, png_filename):
    # Define bin edges: from 0.1 to 0.9 (in steps of 0.1) and last bin is [0.9, ∞)
    bin_edges = np.array(list(np.arange(0.1, 0.9, 0.1)) + [0.9, np.inf])
    # Get bin intervals and create custom labels: for last bin, use ">=0.9"
    bin_intervals = pd.cut(df["prob"], bins=bin_edges, right=False).dtype.categories
    bin_labels = [f"{interval.left:.1f}-{interval.right:.1f}" if interval.right != np.inf 
                  else f">={interval.left:.1f}" for interval in bin_intervals]
    x = np.arange(len(bin_labels))
    
    # Compute stats for each bin
    df_copy = df.copy()
    df_copy["prob_bin"] = pd.cut(df_copy["prob"], bins=bin_edges, right=False)
    grouped = df_copy.dropna(subset=["prob_bin"]).groupby("prob_bin")
    stats = grouped["is_correct"].agg(["count", "sum", "mean"])
    stats.rename(columns={"count": "N_in_bin", "sum": "N_correct", "mean": "fraction_correct"}, inplace=True)
    stats = stats.reindex(bin_intervals, fill_value=0)
    fraction_values = stats["fraction_correct"].tolist()
    
    plt.figure(figsize=(18, 6))
    plt.bar(x, fraction_values, width=0.8, color="green", alpha=0.7, edgecolor="black")
    
    # For each bin, add two text labels:
    # Upper text shows the fraction (e.g. "0.85")
    # Lower text shows the count ratio (e.g. "82/100")
    for i in range(len(bin_labels)):
        frac_val = fraction_values[i]
        n_corr = stats["N_correct"].iloc[i]
        n_total = stats["N_in_bin"].iloc[i]
        plt.text(x[i], frac_val + 0.12, f"{frac_val:.2f}", ha="center", va="bottom", fontsize=12)
        plt.text(x[i], frac_val + 0.02, f"{int(n_corr)}/{int(n_total)}", ha="center", va="bottom", fontsize=12)
    
    plt.xlabel("FB Probability Bin", fontsize=16)
    plt.ylabel("Fraction Correct", fontsize=16)
    plt.ylim(0, 1.1)
    plt.xticks(x, bin_labels, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    
    # Set the title based on dataset and category
    if dataset == "t":
        gene_info = "(#TestGene = 1117)"
        ds_title = "Arabidopsis"
    elif dataset == "z":
        gene_info = "(#TestGene = 825)"
        ds_title = "Zebrafish"
    elif dataset == "m":
        gene_info = "(#TestGene = 1212)"
        ds_title = "Mouse"
    elif dataset == "h":
        gene_info = "(#TestGene = 1629)"
        ds_title = "Human"
    else:
        gene_info = ""
        ds_title = dataset.upper()
    
    if category == "5SS":
        plt.title(f"{ds_title} 5SS_top_{top_k} {gene_info}", fontsize=18)
    elif category == "3SS":
        plt.title(f"{ds_title} 3SS_top_{top_k} {gene_info}", fontsize=18)
    else:
        plt.title(f"{ds_title} Combined_top_{top_k} {gene_info}", fontsize=18)
    
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150, bbox_inches="tight")
    print(f"Histogram saved to {png_filename}")
    plt.close()

###############################################################################
# 3) Main code: Process files and generate three PNG histograms
###############################################################################

def main():
    # Choose the input file pattern based on dataset and top_k value
    if dataset == "t":
        pattern = f"./0_t_result/t_result_{top_k}/000_arabidopsis_g_*.txt"
    elif dataset == "z":
        pattern = f"./0_z_result/z_result_{top_k}/000_zebrafish_g_*.txt"
    elif dataset == "m":
        pattern = f"./0_m_result/m_result_{top_k}/000_mouse_g_*.txt"
    elif dataset == "h":
        pattern = f"./0_h_result/h_result_{top_k}/000_human_g_*.txt"
    else:
        print("Unknown dataset")
        return
    
    files = sorted(glob.glob(pattern))
    num_files = len(files)
    print(f"Found {num_files} txt files matching the pattern '{pattern}'.")
    
    if num_files == 0:
        print("No matching files found!")
        return

    all_data = []
    for i, fname in enumerate(files, start=1):
        print(f"\n=== Processing file {i}/{num_files}: {fname} ===")
        df_splice = parse_splice_file(fname)
        all_data.append(df_splice)
    
    if not all_data:
        print("No valid data found in the files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    df_5SS = combined_df[combined_df["type"] == "5prime"]
    df_3SS = combined_df[combined_df["type"] == "3prime"]

    # Save three separate PNG files:
    # 1. For 5SS only
    plot_histogram("5SS", dataset, top_k, df_5SS, f"./0_{dataset}_result/FBP_top_{top_k}_5SS.png")
    # 2. For 3SS only
    plot_histogram("3SS", dataset, top_k, df_3SS, f"./0_{dataset}_result/FBP_top_{top_k}_3SS.png")
    # 3. For combined (5SS+3SS)
    plot_histogram("ALL", dataset, top_k, combined_df, f"./0_{dataset}_result/FBP_top_{top_k}_ALL.png")

if __name__ == "__main__":
    main()
