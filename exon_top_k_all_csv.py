#!/usr/bin/env python
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

trace = False

# ====== Helper functions: from your original script ======
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
        print("annotated_5prime = ", annotated_5prime)
        print("annotated_3prime = ", annotated_3prime)
        print("viterbi_5prime = ", viterbi_5prime)
        print("viterbi_3prime = ", viterbi_3prime)


    pattern_5prime_block = re.compile(r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", re.DOTALL)
    pattern_3prime_block = re.compile(r"Sorted 3['′] Splice Sites .*?\n(.*)", re.DOTALL)
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

    for pos in annotated_5prime - {r[0] for r in rows if r[2] == "5prime"}:
        rows.append((pos, 0.0, "5prime", True, pos in viterbi_5prime))
    for pos in annotated_3prime - {r[0] for r in rows if r[2] == "3prime"}:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - {r[0] for r in rows if r[2] == "5prime"}:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - {r[0] for r in rows if r[2] == "3prime"}:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])

def prob3SS(pos, df, method="annotated"):
    row = df[(df["type"] == "3prime") & (df[method_column(method)] == True) & (df["position"] == pos)]
    return row.iloc[0]["prob"] if not row.empty else 0

def prob5SS(pos, df, method="annotated"):
    row = df[(df["type"] == "5prime") & (df[method_column(method)] == True) & (df["position"] == pos)]
    return row.iloc[0]["prob"] if not row.empty else 0

def method_column(method):
    return "is_correct" if method == "annotated" else "is_viterbi"

# ====== Main batch processing ======
species_map = {
    't': 'arabidopsis',
    'z': 'zebrafish',
    'm': 'mouse',
    'h': 'human',
    'f': 'fly',
    'o': 'moth'
}

seeds = [0, 1, 2]
top_ks = [100, 500, 1000]
# top_ks = [100]
results = []

for seed in tqdm(seeds):
    for top_k in tqdm(top_ks):
        for code, name in species_map.items():
            pattern = f"./FBP_SMsplice/{seed}_t_z_m_h_f_o_result/{seed}_{code}_result/{code}_result_{top_k}/000_{name}_g_*.txt"
            file_list = glob.glob(pattern)
            if not file_list:
                print(f"No files for {code}-{seed}-{top_k}")
                continue

            correct_exon_scores = []
            incorrect_exon_scores = []

            for txt_file in file_list:
                with open(txt_file) as f:
                    content = f.read()

                match = re.search(r"Annotated 5SS:\s*(\[[^\]]*\])", content)
                if not match: continue
                ann5ss = list(map(int, re.findall(r'\d+', match.group(1))))
                match = re.search(r"Annotated 3SS:\s*(\[[^\]]*\])", content)
                if not match: continue
                ann3ss = list(map(int, re.findall(r'\d+', match.group(1))))
                match = re.search(r"SMsplice 5SS:\s*(\[[^\]]*\])", content)
                if not match: continue
                sm5ss = list(map(int, re.findall(r'\d+', match.group(1))))
                match = re.search(r"SMsplice 3SS:\s*(\[[^\]]*\])", content)
                if not match: continue
                sm3ss = list(map(int, re.findall(r'\d+', match.group(1))))


                ann_pairs = [(0, ann5ss[0])] + [(ann3ss[i], ann5ss[i+1]) for i in range(min(len(ann3ss), len(ann5ss)-1))] + [(ann3ss[-1], -1)]
                sm_pairs = [(0, sm5ss[0])] + [(sm3ss[i], sm5ss[i+1]) for i in range(min(len(sm3ss), len(sm5ss)-1))] + [(sm3ss[-1], -1)]

                if trace:
                    print("ann_pairs = ", ann_pairs)
                    print("sm_pairs = ", sm_pairs)

                df = parse_splice_file(txt_file)
                for a, b in sm_pairs:
                    p3 = prob3SS(a, df, method="smsplice")
                    p5 = prob5SS(b, df, method="smsplice")
                    score = p3 * p5

                    if (a == 0) and (b in ann5ss):
                        p3 = 1
                        score = p3 * p5
                        correct_exon_scores.append(score)
                        if trace:
                            print(f"Exon {a, b}: prob3SS({a}) = {p3}, prob5SS({b}) = {p5}, exon_score = {score}")
                    elif (a in ann3ss) and (b == -1):
                        p5 = 1
                        score = p3 * p5
                        correct_exon_scores.append(score)
                        if trace:
                            print(f"Exon {a, b}: prob3SS({a}) = {p3}, prob5SS({b}) = {p5}, exon_score = {score}")


                    elif (a in ann3ss) and (b in ann5ss):
                        correct_exon_scores.append(score)
                        if trace:
                            print(f"Exon {a, b}: prob3SS({a}) = {p3}, prob5SS({b}) = {p5}, exon_score = {score}")
                    else:

                        if a == 0:
                            p3 = 1
                            score = p3 * p5   
                        if b == -1:
                            p5 = 1
                            score = p3 * p5              

                        incorrect_exon_scores.append(score)
                        if trace:
                            print(f"Exon {a, b}: prob3SS({a}) = {p3}, prob5SS({b}) = {p5}, exon_score = {score}")

                    

            correct_mean = np.mean(correct_exon_scores) if correct_exon_scores else None
            incorrect_mean = np.mean(incorrect_exon_scores) if incorrect_exon_scores else None
            results.append({
                "seed": seed,
                "top_k": top_k,
                "species": code,
                "correct_mean": correct_mean,
                "incorrect_mean": incorrect_mean
            })

# ====== Save result as CSV ======
df_result = pd.DataFrame(results)
df_result.to_csv("smsplice_exon_scores_top_k.csv", index=False)
print("Saved results to smsplice_exon_scores_top_k.csv")
