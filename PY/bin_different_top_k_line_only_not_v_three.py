#!/usr/bin/env python3
import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

dataset = sys.argv[1]  # e.g. "t", "z", "m", or "h"

def parse_splice_file(filename):
    """Parse a .txt file to get a DataFrame with columns:
       [position, prob, type, is_correct, is_viterbi]."""
    with open(filename, "r") as f:
        text = f.read()

    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
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
        return set(map(int, items)) if items != [''] else set()

    annotated_5prime = parse_list(pattern_5ss)
    annotated_3prime = parse_list(pattern_3ss)
    viterbi_5prime = parse_list(pattern_smsplice_5ss)
    viterbi_3prime = parse_list(pattern_smsplice_3ss)

    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
        re.DOTALL)
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)",
        re.DOTALL)
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

    df = pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])
    return df

def plot_multi_topk_desc_ecdf_with_line_and_stacked_labels(dataset, topk_list):
    """Plots Non-Viterbi fraction lines (cumulative correct fraction) and
       also places the 'count/total' text below the x-axis."""
    categories = [("5SS", "5prime"), ("3SS", "3prime"), ("ALL", "ALL")]

    # Colors for each top_k
    nonvit_colors = {1000: "red", 500: "blue", 100: "green"}

    # Thresholds from 0.9 down to 0.1
    thresholds = np.arange(0.9, 0.0, -0.1)
    x_positions = np.arange(len(thresholds))  # 0,1,2,... for each threshold

    # We'll define a negative y-offset for each top_k so they don't overlap
    y_offsets = {1000: -0.20, 500: -0.15, 100: -0.10}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, (cat_label, cat_filter) in zip(axes, categories):
        # We'll track the maximum fraction so we can set a good y-limit
        max_frac = 0.0

        # We'll add dummy lines for the legend
        for tk in topk_list:
            ax.plot([], [], color=nonvit_colors[tk], label=f"top_k={tk}")

        for top_k_val in topk_list:
            if dataset == "t":
                pattern = f"./0_t_result/t_result_{top_k_val}/000_arabidopsis_g_*.txt"
            elif dataset == "z":
                pattern = f"./0_z_result/z_result_{top_k_val}/000_zebrafish_g_*.txt"
            elif dataset == "m":
                pattern = f"./0_m_result/m_result_{top_k_val}/000_mouse_g_*.txt"
            elif dataset == "h":
                pattern = f"./0_h_result/h_result_{top_k_val}/000_human_g_*.txt"
            else:
                print("Unknown dataset")
                return

            files = sorted(glob.glob(pattern))
            if not files:
                print(f"No files found for top_k = {top_k_val}")
                continue

            all_data = [parse_splice_file(fname) for fname in files]
            combined_df = pd.concat(all_data, ignore_index=True)

            # Filter category
            if cat_filter != "ALL":
                df_cat = combined_df[combined_df["type"] == cat_filter]
            else:
                df_cat = combined_df

            # Only Non-Viterbi group
            df_nonvit = df_cat[~df_cat["is_viterbi"]]

            # Helper: returns (fraction, correct, total)
            def cum_stats(df, threshold):
                subset = df[df["prob"] >= threshold]
                total = len(subset)
                if total == 0:
                    return (np.nan, 0, 0)
                correct = subset["is_correct"].sum()
                frac = correct / total
                return (frac, correct, total)

            results = [cum_stats(df_nonvit, t) for t in thresholds]
            fractions = [r[0] for r in results]
            # We'll track max fraction
            cur_max = np.nanmax(fractions)
            if not np.isnan(cur_max):
                max_frac = max(max_frac, cur_max)

            # PLOT the fraction line
            ax.plot(x_positions, fractions, marker='o', color=nonvit_colors[top_k_val])

            # Place the text below the x-axis
            offset_y = y_offsets[top_k_val]
            for i, (frac, corr, tot) in enumerate(results):
                if not np.isnan(frac):
                    text_str = f"{corr}/{tot}"
                    ax.text(i, offset_y, text_str,
                            ha="center", va="top",
                            color=nonvit_colors[top_k_val],
                            fontsize=9,
                            transform=ax.get_xaxis_transform())

        # X-axis ticks
        bin_labels = [f"{thresholds[i]:.1f}-{thresholds[i]+0.1:.1f}" for i in range(len(thresholds))]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels)
        # Invert x-axis so 0.9-1.0 is on the left
        ax.invert_xaxis()
        # Give some space for negative offsets
        ax.set_ylim(0, max(1.05, max_frac + 0.05))
        ax.grid(True)
        # ax.set_xlabel("Lower Bound of FB Probability Bin", fontsize=12)
        ax.set_title(cat_label, fontsize=14)
        ax.legend(fontsize=9)
        ax.invert_xaxis()



    axes[0].set_ylabel("Cumulative Correct Fraction", fontsize=12)
    fig.suptitle(f"Non-Viterbi (dataset={dataset})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpng = f"./0_{dataset}_result/not_v_{dataset}.png"
    plt.savefig(outpng, dpi=150, bbox_inches="tight")
    print(f"Saved to {outpng}")
    plt.close()

def main():
    # We'll compare top_k=100, 500, 1000
    topk_list = [100, 500, 1000]
    plot_multi_topk_desc_ecdf_with_line_and_stacked_labels(dataset, topk_list)

if __name__ == "__main__":
    main()
