#!/usr/bin/env python3

import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Get dataset identifier and top_k value from command-line arguments
# Usage: python3 plot_fp_fn_tp_boxplot_bw.py t 100
dataset = sys.argv[1]  # e.g., "t", "z", "m", or "h"
top_k = int(sys.argv[2])

###############################################################################
# Helper function: set title based on dataset and category
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
        return f"{category}: FP, TP, FN, & FP+FN Distributions ({dataset.upper()})"

###############################################################################
# Function to record metric distributions and computed scores to a text file
###############################################################################
def record_metric_distributions(distributions, output_filename):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        for cat, metrics in distributions.items():
            f.write(f"Category: {cat}\n")
            for metric, values in metrics.items():
                f.write(f"  {metric}: {values}\n")
            # Compute counts and scores
            tp_count = len(metrics["TP"])
            fp_count = len(metrics["FP"])
            fn_count = len(metrics["FN"])
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f.write(f"  TP_count: {tp_count}\n")
            f.write(f"  FP_count: {fp_count}\n")
            f.write(f"  FN_count: {fn_count}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write("\n")
    print(f"Metric distributions and scores saved to {output_filename}")

###############################################################################
# Function to parse splice site file and extract SMsplice (Viterbi) information.
# For each annotated splice site or each SMsplice (Viterbi) prediction that is missing 
# from the sorted predictions, a row is added with probability 0.
###############################################################################
def parse_splice_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Regular expressions for annotated splice sites
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    # Regular expressions for SMsplice (Viterbi) splice sites
    pattern_smsplice_5ss = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
    pattern_smsplice_3ss = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")

    def parse_list(regex):
        match = regex.search(text)
        if not match:
            return set()
        inside = match.group(1).strip()
        if not inside:
            return set()
        items = re.split(r"[\s,]+", inside.strip())
        return set(map(int, items))
    
    annotated_5prime = parse_list(pattern_5ss)
    annotated_3prime = parse_list(pattern_3ss)
    viterbi_5prime = parse_list(pattern_smsplice_5ss)
    viterbi_3prime = parse_list(pattern_smsplice_3ss)

    # Regular expressions for sorted prediction blocks
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
        in_viterbi = (pos in viterbi_5prime)
        rows.append((pos, prob, "5prime", is_correct, in_viterbi))
    for (pos, prob) in threeprime_preds:
        is_correct = (pos in annotated_3prime)
        in_viterbi = (pos in viterbi_3prime)
        rows.append((pos, prob, "3prime", is_correct, in_viterbi))
    
    # Add missing annotated splice sites (for FN) using correct viterbi flag.
    existing_5prime = set([row[0] for row in rows if row[2] == "5prime"])
    for pos in annotated_5prime:
        if pos not in existing_5prime:
            rows.append((pos, 0.0, "5prime", True, (pos in viterbi_5prime)))
    
    existing_3prime = set([row[0] for row in rows if row[2] == "3prime"])
    for pos in annotated_3prime:
        if pos not in existing_3prime:
            rows.append((pos, 0.0, "3prime", True, (pos in viterbi_3prime)))
    
    # Also add missing viterbi predictions (for FP) that are not annotated.
    # (These are false positives that are in the viterbi set but missing from the sorted predictions.)
    existing_5prime = set([row[0] for row in rows if row[2] == "5prime"])
    for pos in viterbi_5prime:
        if pos not in existing_5prime:
            # For a non-annotated site, is_correct is False.
            rows.append((pos, 0.0, "5prime", False, True))
    
    existing_3prime = set([row[0] for row in rows if row[2] == "3prime"])
    for pos in viterbi_3prime:
        if pos not in existing_3prime:
            rows.append((pos, 0.0, "3prime", False, True))
    
    df = pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])
    return df

###############################################################################
# Function to get metric distributions from a DataFrame.
# Returns a dictionary with keys: 'TP', 'FP', 'FN', 'FP+FN'
###############################################################################
def get_metric_distributions(df):
    # True Positive: SMsplice predictions that are correct.
    tp_vals = df[(df["is_viterbi"].astype(bool)) & (df["is_correct"])]['prob'].tolist()
    # False Positive: SMsplice predictions that are incorrect.
    fp_vals = df[(df["is_viterbi"].astype(bool)) & (~df["is_correct"])]['prob'].tolist()
    # False Negative: Non-SMsplice predictions that are correct.
    fn_vals = df[(~df["is_viterbi"].astype(bool)) & (df["is_correct"])]['prob'].tolist()
    # Combined FP+FN.
    combined_vals = fp_vals + fn_vals
    return {"TP": tp_vals, "FP": fp_vals, "FN": fn_vals, "FP+FN": combined_vals}

###############################################################################
# Function to plot box plots for FP, TP, FN, and FP+FN distributions in black & white,
# with no hatching and red mean markers.
# X-axis tick labels now include only the total count for each metric.
###############################################################################
def plot_fp_fn_tp_boxplots(metric_dict, output_filename):
    categories = ["5SS", "3SS", "ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # Use a custom meanprops to make the mean markers red
    meanprops = dict(marker='^', markerfacecolor='red', markeredgecolor='red')

    for i, cat in enumerate(categories):
        ax = axes[i]
        metrics = metric_dict[cat]
        # Order: TP, FP, FN, FP+FN.
        data = [metrics["TP"], metrics["FP"], metrics["FN"], metrics["FP+FN"]]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanprops=meanprops)
        
        # Make all boxes white with black edges, no hatch
        for box in bp['boxes']:
            box.set_facecolor("white")
            box.set_edgecolor("black")
        # Whiskers, caps, medians: black lines
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color("black")
        
        # Compute counts for annotation.
        tp_count = len(metrics["TP"])
        fp_count = len(metrics["FP"])
        fn_count = len(metrics["FN"])
        fp_fn_count = fp_count + fn_count

        # Create custom x-axis tick labels with total count only.
        labels = [
            f"TP (n={tp_count})",
            f"FP (n={fp_count})",
            f"FN (n={fn_count})",
            f"FP+FN (n={fp_fn_count})"
        ]
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("FBP", fontsize=14)
        ax.set_title(get_title(cat), fontsize=16)
    
    plt.suptitle(f"FP, TP, FN, and FP+FN Probability Distributions (top_k = {top_k})", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Combined box plot saved to {output_filename}")
    plt.close()

###############################################################################
# New Function: Plot separate boxplots for each metric across categories.
# For each metric (TP, FP, FN, FP+FN), a subplot is created showing the distribution
# for the categories: 5SS, 3SS, and ALL.
###############################################################################
def plot_boxplots_by_metric(metric_dict, output_filename):
    metrics_list = ["TP", "FP", "FN", "FP+FN"]
    categories = ["5SS", "3SS", "ALL"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()
    meanprops = dict(marker='^', markerfacecolor='red', markeredgecolor='red')
    
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        data = []
        labels = []
        # For each category, get the distribution for this metric
        for cat in categories:
            vals = metric_dict[cat][metric]
            data.append(vals)
            count = len(vals)
            labels.append(f"{cat} (n={count})")
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanprops=meanprops)
        for box in bp['boxes']:
            box.set_facecolor("white")
            box.set_edgecolor("black")
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color("black")
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric, fontsize=16)
        ax.set_ylabel("FBP", fontsize=14)
    
    plt.suptitle(f"Boxplots by Metric (top_k = {top_k})", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Boxplots by metric saved to {output_filename}")
    plt.close()

###############################################################################
# New Function: Plot precision, recall, and F1 scores across categories.
# A grouped bar chart is created for the three scores for each category.
###############################################################################
def plot_score_metrics(metric_dict, output_filename):
    categories = ["5SS", "3SS", "ALL"]
    precision_vals = []
    recall_vals = []
    f1_vals = []
    for cat in categories:
        metrics = metric_dict[cat]
        tp_count = len(metrics["TP"])
        fp_count = len(metrics["FP"])
        fn_count = len(metrics["FN"])
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
    
    x = np.arange(len(categories))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width, precision_vals, width, label='Precision')
    rects2 = ax.bar(x, recall_vals, width, label='Recall')
    rects3 = ax.bar(x + width, f1_vals, width, label='F1 Score')
    ax.set_ylabel('Score')
    ax.set_title(f'Precision, Recall, and F1 Score (top_k = {top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    # Annotate bar values
    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Score metrics plot saved to {output_filename}")
    plt.close()

###############################################################################
# Main function: process files, compute metric distributions,
# generate box plots, record distributions & scores, and plot score metrics.
###############################################################################
def main():
    # Select file pattern based on dataset
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
    print(f"Found {num_files} txt files matching pattern '{pattern}'.")
    if num_files == 0:
        print("No files found!")
        return

    all_data = []
    for i, fname in enumerate(files, start=1):
        print(f"\n=== Processing file {i}/{num_files}: {fname} ===")
        df_splice = parse_splice_file(fname)
        all_data.append(df_splice)
    
    if not all_data:
        print("No valid data found in files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    df_5SS = combined_df[combined_df["type"] == "5prime"]
    df_3SS = combined_df[combined_df["type"] == "3prime"]

    # Compute metric distributions for each category
    metric_5SS = get_metric_distributions(df_5SS)
    metric_3SS = get_metric_distributions(df_3SS)
    metric_ALL = get_metric_distributions(combined_df)
    
    metric_dict = {
        "5SS": metric_5SS,
        "3SS": metric_3SS,
        "ALL": metric_ALL
    }
    
    # Plot box plots for FP, TP, FN, and FP+FN distributions by category
    output_png = f"./0_{dataset}_result/fp_fn_tp_boxplot_bw_top_{top_k}_{dataset}.png"
    plot_fp_fn_tp_boxplots(metric_dict, output_png)
    
    # Plot additional boxplots by metric across categories
    output_png_by_metric = f"./0_{dataset}_result/boxplots_by_metric_top_{top_k}_{dataset}.png"
    plot_boxplots_by_metric(metric_dict, output_png_by_metric)
    
    # Plot precision, recall, and F1 scores across categories
    output_png_scores = f"./0_{dataset}_result/score_metrics_top_{top_k}_{dataset}.png"
    plot_score_metrics(metric_dict, output_png_scores)
    
    # Record the metric distributions and computed scores to a text file
    output_txt = f"./0_{dataset}_result/fp_fn_tp_boxplot_bw_top_{top_k}_{dataset}.txt"
    record_metric_distributions(metric_dict, output_txt)
    
if __name__ == "__main__":
    main()
