#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time, argparse, pickle, re, sys, math
import matplotlib.pyplot as plt
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
from SMsplice import *  # Assumes viterbi_FBP is defined here
from ipdb import set_trace

# --------------------------
# Load data from pickle file
# --------------------------

dataset = sys.argv[1]
top_k = int(sys.argv[2])

para_values = [0.9, 0.8, 0.7, 0.6]


if dataset == "t":
    with open('t_new.pkl', 'rb') as f:
        data = pickle.load(f)
elif dataset == "z":
    with open('z_new.pkl', 'rb') as f:
        data = pickle.load(f)
elif dataset == "m":
    with open('m_new.pkl', 'rb') as f:
        data = pickle.load(f)
elif dataset == "h":
    with open('h_new.pkl', 'rb') as f:
        data = pickle.load(f)

# For multiple runs over different para thresholds, define a range.
# (Adjust these as needed.)
# para_values = np.linspace(0.02, 0.03, 3)


# Baseline F1 from the run at para = 0.0 (given)
if dataset == "t":
    baseline_f1 = 0.8530500705218619
elif dataset == "z":
    baseline_f1 = 0.859149111937216
elif dataset == "m":
    baseline_f1 = 0.7285912974369163
elif dataset == "h":
    baseline_f1 = 0.7018601446779194

# --------------------------
# Extract variables from pickle
# --------------------------
sequences = data['sequences']
pME = data['pME']
pELF = data['pELF']
pIL = data['pIL']
pEE = data['pEE']
pELM = data['pELM']
pEO = data['pEO']
pELL = data['pELL']
# Make copies of the original emissions so that we can reinitialize for each run
emissions5_orig = data['emissions5']
emissions3_orig = data['emissions3']
lengths = data['lengths']
trueSeqs = data['trueSeqs']
testGenes = data['testGenes']
B3 = data['B3']
B5 = data['B5']

# Determine the number of genes to process based on the dataset
if dataset == "t":
    len_index = 1117
elif dataset == "z":
    len_index = 825
elif dataset == "m":
    len_index = 1212
elif dataset == "h":
    len_index = 1629
else:
    print("Dataset not recognized.")
    sys.exit(1)

# Lists to store results for each para value
results_para = []
results_recall = []
results_precision = []
results_f1 = []
results_improvement = []

# --------------------------
# Define helper functions
# --------------------------
def parse_list(regex, text):
    match = regex.search(text)
    if not match:
        return set()
    inside = match.group(1).strip()
    if not inside:
        return set()
    items = re.split(r"[\s,]+", inside)
    return set(map(int, items))

def parse_predictions(pattern, text):
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")
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

# --------------------------
# Loop over different para thresholds
# --------------------------
for para in para_values:
    # Reset emissions for each run (use a deep copy so original values remain unchanged)
    emissions5 = emissions5_orig.copy()
    emissions3 = emissions3_orig.copy()
    
    # Prepare arrays for storing probabilities (not used later for metrics, but for completeness)
    fbp5 = emissions5.copy()
    fbp3 = emissions3.copy()
    fbp5[:] = -np.inf
    fbp3[:] = -np.inf

    # Process each gene file (for indices 0 to len_index-1)
    for i in range(0, len_index):
        if dataset == "t":
            input_filename = f"./0_t_result/t_result_{top_k}/000_arabidopsis_g_{i}.txt"
        elif dataset == "z":
            input_filename = f"./0_z_result/z_result_{top_k}/000_zebrafish_g_{i}.txt"
        elif dataset == "m":
            input_filename = f"./0_m_result/m_result_{top_k}/000_mouse_g_{i}.txt"
        elif dataset == "h":
            input_filename = f"./0_h_result/h_result_{top_k}/000_human_g_{i}.txt"

        with open(input_filename, "r") as f:
            text = f.read()

        # Define regex patterns to parse annotated and predicted splice sites
        pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
        pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
        pattern_smsplice_5ss = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
        pattern_smsplice_3ss = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")

        annotated_5prime = parse_list(pattern_5ss, text)
        annotated_3prime = parse_list(pattern_3ss, text)
        viterbi_5prime = parse_list(pattern_smsplice_5ss, text)
        viterbi_3prime = parse_list(pattern_smsplice_3ss, text)

        # Patterns to parse sorted predictions
        pattern_5prime_block = re.compile(
            r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
            re.DOTALL
        )
        pattern_3prime_block = re.compile(
            r"Sorted 3['′] Splice Sites .*?\n(.*)",
            re.DOTALL
        )

        fiveprime_preds = parse_predictions(pattern_5prime_block, text)
        threeprime_preds = parse_predictions(pattern_3prime_block, text)

        # Update fbp and emissions arrays based on the current para threshold
        for pos, prob in fiveprime_preds:
            fbp5[i][pos] = prob
            if prob < para:
                emissions5[i][pos] = -np.inf

        for pos, prob in threeprime_preds:
            fbp3[i][pos] = prob
            if prob < para:
                emissions3[i][pos] = -np.inf

    # --------------------------
    # Run the Viterbi algorithm using the updated emissions arrays.
    # --------------------------
    pred_all = viterbi_FBP(
        sequences=sequences,
        emissions5=emissions5,
        emissions3=emissions3,
        pME=pME,
        pEE=pEE,
        pEO=pEO,
        pIL=pIL,
        pELF=pELF,
        pELM=pELM,
        pELL=pELL
    )

    # --------------------------
    # Compute Sensitivity, Precision, and F1-score across genes.
    # --------------------------
    num_truePositives = 0
    num_falsePositives = 0
    num_falseNegatives = 0

    predFives_all = []
    predThrees_all = []
    trueFives_all = []
    trueThrees_all = []

    for g, gene in enumerate(testGenes):
        L = lengths[g]
        predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
        predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

        predFives_all.append(predFives)
        predThrees_all.append(predThrees)
        trueFives_all.append(trueFives)
        trueThrees_all.append(trueThrees)

        num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))
        num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(np.setdiff1d(predFives, trueFives))
        num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(np.setdiff1d(trueFives, predFives))

    ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
    ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
    improvement = 100 * (f1 - baseline_f1)

    # Print the metrics for the current para
    print(f"para = {para:.3f}, Recall: {ssSens:.4f}, Precision: {ssPrec:.4f}, F1: {f1:.4f}")
    print(f"F1 improved by {improvement:.4f}%\n")
    
    # Save results
    results_para.append(para)
    results_recall.append(ssSens)
    results_precision.append(ssPrec)
    results_f1.append(f1)
    results_improvement.append(improvement)

# --------------------------
# Plot the metrics versus the threshold (para)
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(results_para, results_recall, marker='o', label='Recall')
plt.plot(results_para, results_precision, marker='o', label='Precision')
plt.plot(results_para, results_f1, marker='o', label='F1')

plt.xlabel('Parameter (para)')
plt.ylabel('Metric Value')
plt.title('Metrics vs. Parameter Threshold (para)')

# Mark the point with maximum F1 improvement
max_idx = np.argmax(results_improvement)
best_para = results_para[max_idx]
best_improvement = results_improvement[max_idx]
best_f1 = results_f1[max_idx]

plt.scatter([best_para], [best_f1], color='red', zorder=5)
plt.axvline(x=best_para, color='red', linestyle='--', label=f'Best para: {best_para:.3f}')
plt.annotate(f'Max improvement: {best_improvement:.2f}%\n(para = {best_para:.3f})',
             xy=(best_para, best_f1),
             xytext=(best_para + 0.02, best_f1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()
plt.savefig(f"0_metrics_vs_para_{dataset}_{top_k}.png")
