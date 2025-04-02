#!/usr/bin/env python
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import seaborn as sns
from ipdb import set_trace

# Set to True for verbose output
trace = True

# ====== Seaborn 風格設定 ======
sns.set_style("whitegrid")
sns.set_context("notebook")

# ====== Helper functions ======
def parse_splice_file(filename):
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
        return set(map(int, items))

    annotated_5prime = parse_list(pattern_5ss)
    annotated_3prime = parse_list(pattern_3ss)
    viterbi_5prime = parse_list(pattern_smsplice_5ss)
    viterbi_3prime = parse_list(pattern_smsplice_3ss)

    if trace:
        print("annotated_5prime =", annotated_5prime)
        print("annotated_3prime =", annotated_3prime)
        print("viterbi_5prime =", viterbi_5prime)
        print("viterbi_3prime =", viterbi_3prime)

    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)", re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(pattern):
        match_block = pattern.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        return [(int(m.group(1)), float(m.group(2))) for m in pattern_line.finditer(block)]

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    rows = []
    for (pos, prob) in fiveprime_preds:
        rows.append((pos, prob, "5prime", pos in annotated_5prime, pos in viterbi_5prime))
    for (pos, prob) in threeprime_preds:
        rows.append((pos, prob, "3prime", pos in annotated_3prime, pos in viterbi_3prime))

    # Add missing predictions (prob=0) if they appear in annotated or viterbi sets
    for pos in annotated_5prime - {r[0] for r in rows if r[2] == "5prime"}:
        rows.append((pos, 0.0, "5prime", True, pos in viterbi_5prime))
    for pos in annotated_3prime - {r[0] for r in rows if r[2] == "3prime"}:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - {r[0] for r in rows if r[2] == "5prime"}:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - {r[0] for r in rows if r[2] == "3prime"}:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])

def method_column(method):
    return "is_correct" if method == "annotated" else "is_viterbi"

def prob3SS(pos, df, method="annotated"):
    row = df[(df["type"]=="3prime") & (df[method_column(method)]==True) & (df["position"]==pos)]
    return row.iloc[0]["prob"] if not row.empty else 0

def prob5SS(pos, df, method="annotated"):
    row = df[(df["type"]=="5prime") & (df[method_column(method)]==True) & (df["position"]==pos)]
    return row.iloc[0]["prob"] if not row.empty else 0

# ====== Main Batch Processing ======
species_map = {
    't': 'arabidopsis',
    'z': 'zebrafish',
    'm': 'mouse',
    'h': 'human',
    'f': 'fly',
    'o': 'moth'
}

seeds = [0]  # 也可自行擴充為 [0,1,2] 或更多
top_ks = [1000]  # 也可擴充為 [100, 500, 1000] 或更多

results = []
all_intron_scores = []

for seed in tqdm(seeds, desc="Seeds"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, name in species_map.items():
            pattern = (
                f"./{seed}_{code}_intron_score/"
                f"{code}_result_{top_k}/000_{name}_g_*.txt"
            )
            file_list = glob.glob(pattern)
            if not file_list:
                continue

            correct_intron_scores = []
            # Initialize lists for incorrect scores
            incorrect_a_correct = []
            incorrect_b_correct = []
            incorrect_both = []

            for txt_file in file_list:
                with open(txt_file) as f:
                    content = f.read()

                # 解析 Annotated/SMsplice 5SS 與 3SS
                match = re.search(r"Annotated 5SS:\s*(\[[^\]]*\])", content)
                if not match: 
                    continue
                ann5ss = list(map(int, re.findall(r'\d+', match.group(1))))
                
                match = re.search(r"Annotated 3SS:\s*(\[[^\]]*\])", content)
                if not match: 
                    continue
                ann3ss = list(map(int, re.findall(r'\d+', match.group(1))))
                
                match = re.search(r"SMsplice 5SS:\s*(\[[^\]]*\])", content)
                if not match: 
                    continue
                sm5ss = list(map(int, re.findall(r'\d+', match.group(1))))
                
                match = re.search(r"SMsplice 3SS:\s*(\[[^\]]*\])", content)
                if not match: 
                    continue
                sm3ss = list(map(int, re.findall(r'\d+', match.group(1))))

                # 建立 intron pair：配對 5′ 與 3′ splice sites
                ann_introns = [
                    (ann5ss[i], ann3ss[i]) for i in range(min(len(ann5ss), len(ann3ss)))
                ]
                sm_introns = [
                    (sm5ss[i], sm3ss[i]) for i in range(min(len(sm5ss), len(sm3ss)))
                ]

                df_splice = parse_splice_file(txt_file)

                # ---------------------------
                # New: Parse the intron prediction table if available
                intron_table = {}
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    # Changed regex order from "3SS, 5SS, prob" to "5SS, 3SS, prob"
                    if re.search(r"5SS,\s*3SS,\s*prob", line, re.IGNORECASE):
                        # Process subsequent lines until an empty line or end of file
                        for subsequent_line in lines[i+1:]:
                            subsequent_line = subsequent_line.strip()
                            if not subsequent_line:
                                break
                            parts = subsequent_line.split(',')
                            if len(parts) >= 3:
                                try:
                                    intron_5ss = int(parts[0].strip())
                                    intron_3ss = int(parts[1].strip())
                                    intron_prob = float(parts[2].strip())
                                    intron_table[(intron_5ss, intron_3ss)] = intron_prob
                                except Exception as e:
                                    continue
                        break
                # ---------------------------

                for (a, b) in sm_introns:
                    # For introns, a is the 5′ splice site and b is the 3′ splice site.
                    p5 = prob5SS(a, df_splice, method="smsplice")
                    p3 = prob3SS(b, df_splice, method="smsplice")

                    if p5 > 1.001 or p3 > 1.001:
                        set_trace()
                    else:
                        print("++++++++++++++++++++++++++++++++")

                    # Use the intron table score if available; otherwise, use the product p5*p3
                    if (a, b) in intron_table:
                        score = intron_table[(a, b)]
                    else:
                        score = p5 * p3

                    if score == 0.0:
                        continue

                    a_correct_flag = (a in ann5ss)
                    b_correct_flag = (b in ann3ss)

                    if a_correct_flag and b_correct_flag:
                        correct_intron_scores.append(score)
                        all_intron_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'correct',
                            'score': score
                        })
                    else:
                        all_intron_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'incorrect',
                            'score': score
                        })
                        if a_correct_flag and not b_correct_flag:
                            incorrect_a_correct.append(score)
                        elif not a_correct_flag and b_correct_flag:
                            incorrect_b_correct.append(score)
                        else:
                            incorrect_both.append(score)

            # 摘要統計
            overall_incorrect = incorrect_a_correct + incorrect_b_correct + incorrect_both
            correct_mean = np.mean(correct_intron_scores) if correct_intron_scores else None
            incorrect_mean = np.mean(overall_incorrect) if overall_incorrect else None

            results.append({
                "seed": seed,
                "top_k": top_k,
                "species": code,
                "correct_mean": correct_mean,
                "incorrect_mean": incorrect_mean,
                "incorrect_a_correct_mean": np.mean(incorrect_a_correct) if incorrect_a_correct else None,
                "incorrect_b_correct_mean": np.mean(incorrect_b_correct) if incorrect_b_correct else None,
                "incorrect_both_mean": np.mean(incorrect_both) if incorrect_both else None
            })

# ====== 儲存摘要 ======
df_result = pd.DataFrame(results)
df_result.to_csv("smsplice_intron_scores_top_k_class.csv", index=False)
print("Saved summary means to smsplice_intron_scores_top_k_class.csv")

# ====== 建立跨所有 seeds 的 Intron Scores DataFrame ======
df_intron_scores = pd.DataFrame(all_intron_scores)
print("\nNumber of total intron predictions recorded:", len(df_intron_scores))
print(df_intron_scores.head())

# -------------------------------------------------------------------------
# PLOTTING SECTION (Aggregate across seeds)
# -------------------------------------------------------------------------
def ecdf(data):
    """回傳用於繪製 eCDF 的 x, y 值。"""
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y

# 針對每個 species，將所有 seed 與 top_k 的資料合併
species_list = df_intron_scores['species'].unique()

for sp in species_list:
    sub_df = df_intron_scores[df_intron_scores['species'] == sp]
    correct_scores = sub_df[sub_df['label'] == 'correct']['score'].values
    incorrect_scores = sub_df[sub_df['label'] == 'incorrect']['score'].values

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # -------------------- 左圖：Boxplot + Swarmplot --------------------
    plot_data = pd.DataFrame({
        "Intron Score": np.concatenate([correct_scores, incorrect_scores]),
        "Prediction Type": ["Correct"] * len(correct_scores) + ["Incorrect"] * len(incorrect_scores)
    })

    # Boxplot
    sns.boxplot(
        x="Prediction Type", 
        y="Intron Score", 
        data=plot_data,
        ax=ax1, 
        showfliers=False, 
        palette="Set3",
        width=0.9,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"}
    )

    # Swarmplot
    sns.swarmplot(
        x="Prediction Type",
        y="Intron Score",
        data=plot_data,
        ax=ax1,
        hue="Prediction Type",
        palette={"Correct": "#1f77b4", "Incorrect": "#ff7f0e"},
        dodge=False,
        size=1.5,
        linewidth=0,
    )

    ax1.set_xlabel("SMsplice Prediction", fontsize=14)
    ax1.set_ylabel("Intron Score", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # -------------------- 右圖：eCDF --------------------
    ax2.grid(True, alpha=0.3)
    n_correct = len(correct_scores)
    n_incorrect = len(incorrect_scores)
    if n_correct > 0:
        x_c, y_c = ecdf(correct_scores)
        ax2.step(x_c, y_c, where='post', label=f"Correct (n={n_correct})")
    if n_incorrect > 0:
        x_i, y_i = ecdf(incorrect_scores)
        ax2.step(x_i, y_i, where='post', label=f"Incorrect (n={n_incorrect})")
    
    ax2.set_xlabel("Intron Score", fontsize=14)
    ax2.set_ylabel("eCDF", fontsize=14)
    ax2.legend(loc='upper left', fontsize=14)
    ax2.tick_params(axis='both', labelsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./0_intron_png/0_{sp}_intron_score_parse_version.png", dpi=300)
    plt.close(fig)

print("\nPlots saved for each species, combining all seeds into one distribution per species.")
