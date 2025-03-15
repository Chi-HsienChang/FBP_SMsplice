#!/usr/bin/env python3

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Get dataset from command-line argument
# Usage: python3 plot_topk_hist.py t
dataset = sys.argv[1]  # e.g., "t", "z", "m", or "h"

# Define list of top_k values; here we use three values: 100, 500, and 1000.
top_k_values = [100, 500, 1000]
# top_k_values = [10]

###############################################################################
# Helper function: Set title based on dataset and category
###############################################################################
def get_title(category):
    # category: "5SS", "3SS", or "ALL"
    if dataset == "t":
        if category == "5SS":
            return "Arabidopsis 5SS (#TestGene = 1117)"
        elif category == "3SS":
            return "Arabidopsis 3SS (#TestGene = 1117)"
        else:
            return "Arabidopsis 5SS+3SS (#TestGene = 1117)"
    elif dataset == "z":
        if category == "5SS":
            return "Zebrafish 5SS (#TestGene = 825)"
        elif category == "3SS":
            return "Zebrafish 3SS (#TestGene = 825)"
        else:
            return "Zebrafish 5SS+3SS (#TestGene = 825)"
    elif dataset == "m":
        if category == "5SS":
            return "Mouse 5SS (#TestGene = 1212)"
        elif category == "3SS":
            return "Mouse 3SS (#TestGene = 1212)"
        else:
            return "Mouse 5SS+3SS (#TestGene = 1212)"
    elif dataset == "h":
        if category == "5SS":
            return "Human 5SS (#TestGene = 1629)"
        elif category == "3SS":
            return "Human 3SS (#TestGene = 1629)"
        else:
            return "Human 5SS+3SS (#TestGene = 1629)"
    else:
        return f"{category}: Fraction Correct vs. FB Probability Bin ({dataset.upper()})"

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
# 2) Compute fraction correct and counts per probability bin from a DataFrame
###############################################################################

def compute_stats_per_bin(df, bin_edges):
    df = df.copy()
    df["prob_bin"] = pd.cut(df["prob"], bins=bin_edges, right=False)
    grouped = df.dropna(subset=["prob_bin"]).groupby("prob_bin")
    stats = grouped["is_correct"].agg(["count", "sum", "mean"])
    stats.rename(columns={"count": "N_in_bin", "sum": "N_correct", "mean": "fraction_correct"}, inplace=True)
    # Ensure all bins (as defined by bin_edges) are present:
    bin_intervals = pd.cut(df["prob"], bins=bin_edges, right=False).dtype.categories
    stats = stats.reindex(bin_intervals, fill_value=0)
    return stats

###############################################################################
# 3) Main loop: Process files, compute separate stats for 5SS, 3SS, and combined,
#    output statistics, and plot grouped bar charts (three PNGs).
###############################################################################

def main():
    # Define bin edges so that bins are: [0.1,0.2), [0.2,0.3), ... [0.8,0.9), and [0.9,∞)
    bin_edges = np.array(list(np.arange(0.1, 0.9, 0.1)) + [0.9, np.inf])
    bin_intervals = pd.cut(pd.Series([0.0]), bins=bin_edges, right=False).dtype.categories
    # Create custom bin labels: if right bound is infinity, label as ">=0.9", otherwise "a-b"
    bin_labels = []
    for interval in bin_intervals:
        if interval.right == np.inf:
            label = f"0.9-1.0"
        else:
            label = f"{interval.left:.1f}-{interval.right:.1f}"
        bin_labels.append(label)
    x = np.arange(len(bin_labels))
    
    # Dictionaries to store stats for 5SS, 3SS, and combined (5SS+3SS)
    stats_dict_5 = {}
    stats_dict_3 = {}
    stats_dict_all = {}
    
    for top_k in top_k_values:
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
        print(f"k = {top_k}: Found {num_files} files with pattern '{pattern}'.")
        if num_files == 0:
            continue
        all_data = []
        for fname in files:
            df_splice = parse_splice_file(fname)
            all_data.append(df_splice)
        if not all_data:
            continue
        combined_df = pd.concat(all_data, ignore_index=True)
        df_5SS = combined_df[combined_df["type"] == "5prime"]
        df_3SS = combined_df[combined_df["type"] == "3prime"]
        stats_5 = compute_stats_per_bin(df_5SS, bin_edges)
        stats_3 = compute_stats_per_bin(df_3SS, bin_edges)
        stats_all = compute_stats_per_bin(combined_df, bin_edges)
        stats_dict_5[top_k] = stats_5
        stats_dict_3[top_k] = stats_3
        stats_dict_all[top_k] = stats_all

    if not stats_dict_5 and not stats_dict_3 and not stats_dict_all:
        print("No data found for any top_k!")
        return

    # Save computed stats for 5SS, 3SS, and combined to text files
    output_txr_5 = f"./0_{dataset}_result/FBP_hist_values_{dataset}_5SS.txt"
    with open(output_txr_5, "w") as f:
        f.write("top_k\tbin_range\tN_correct/N_in_bin\tfraction_correct\n")
        for top_k, stats in stats_dict_5.items():
            for interval, row in stats.iterrows():
                bin_label = f"{interval.left:.1f}-{interval.right:.1f}" if interval.right != np.inf else f">={interval.left:.1f}"
                f.write(f"{top_k}\t{bin_label}\t{int(row['N_correct'])}/{int(row['N_in_bin'])}\t{row['fraction_correct']:.4f}\n")
    print(f"Computed 5SS stats saved to {output_txr_5}")

    output_txr_3 = f"./0_{dataset}_result/FBP_hist_values_{dataset}_3SS.txt"
    with open(output_txr_3, "w") as f:
        f.write("top_k\tbin_range\tN_correct/N_in_bin\tfraction_correct\n")
        for top_k, stats in stats_dict_3.items():
            for interval, row in stats.iterrows():
                bin_label = f"{interval.left:.1f}-{interval.right:.1f}" if interval.right != np.inf else f">={interval.left:.1f}"
                f.write(f"{top_k}\t{bin_label}\t{int(row['N_correct'])}/{int(row['N_in_bin'])}\t{row['fraction_correct']:.4f}\n")
    print(f"Computed 3SS stats saved to {output_txr_3}")

    output_txr_all = f"./0_{dataset}_result/FBP_hist_values_{dataset}_ALL.txt"
    with open(output_txr_all, "w") as f:
        f.write("top_k\tbin_range\tN_correct/N_in_bin\tfraction_correct\n")
        for top_k, stats in stats_dict_all.items():
            for interval, row in stats.iterrows():
                bin_label = f"{interval.left:.1f}-{interval.right:.1f}" if interval.right != np.inf else f">={interval.left:.1f}"
                f.write(f"{top_k}\t{bin_label}\t{int(row['N_correct'])}/{int(row['N_in_bin'])}\t{row['fraction_correct']:.4f}\n")
    print(f"Computed combined stats saved to {output_txr_all}")

    # Define a helper function to plot grouped bar charts
    def plot_grouped_bar_chart(stats_dict, title, output_filename):
        plt.figure(figsize=(35, 8))
        n_topk = len(stats_dict)
        group_width = 0.8
        bar_width = group_width / n_topk
        offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, n_topk)
        for i, tk in enumerate(sorted(stats_dict.keys())):
            stats = stats_dict[tk]
            frac_values = stats["fraction_correct"].tolist()
            # Set color based on top_k value
            if tk == 100:
                color = "green"
            elif tk == 1000:
                color = "red"
            else:
                color = "blue"
            plt.bar(x + offsets[i], frac_values, width=bar_width, color=color, alpha=0.7,
                    edgecolor="black", label=f"k = {tk}")
            n_correct_values = stats["N_correct"].tolist()
            n_total_values = stats["N_in_bin"].tolist()
            for j in range(len(bin_labels)):
                frac_val = frac_values[j]
                # Upper text: the ratio (fraction correct)
                plt.text(x[j] + offsets[i], frac_val + 0.06, f"{frac_val:.5f}",
                         ha="center", va="bottom", fontsize=12)
                # Lower text: the count ratio (N_correct/N_in_bin)
                plt.text(x[j] + offsets[i], frac_val + 0.02, f"{int(n_correct_values[j])}/{int(n_total_values[j])}",
                         ha="center", va="bottom", fontsize=12)
        plt.xlabel("FB Probability Bin", fontsize=24)
        plt.ylabel("Fraction Correct", fontsize=24)
        plt.ylim(0, 1.1)
        plt.xticks(x, bin_labels, fontsize=24, rotation=0)
        plt.yticks(fontsize=24)
        plt.legend(title="top_k", loc="best", fontsize=12, title_fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Histogram saved to {output_filename}")
        plt.close()

    # Plot and save three PNG files:
    # 1. 5SS only
    plot_grouped_bar_chart(stats_dict_5,
                           get_title("5SS"),
                           f"./0_{dataset}_result/FBP_hist_topk_{dataset}_5SS.png")
    # 2. 3SS only
    plot_grouped_bar_chart(stats_dict_3,
                           get_title("3SS"),
                           f"./0_{dataset}_result/FBP_hist_topk_{dataset}_3SS.png")
    # 3. Combined (5SS+3SS)
    plot_grouped_bar_chart(stats_dict_all,
                           get_title("ALL"),
                           f"./0_{dataset}_result/FBP_hist_topk_{dataset}_ALL.png")

if __name__ == "__main__":
    main()
