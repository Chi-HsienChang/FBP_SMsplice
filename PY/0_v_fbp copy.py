import numpy as np
import pandas as pd
import time, argparse#, json, pickle
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
import pickle
from SMsplice import *
import re
import sys
from ipdb import set_trace

# # Load the dictionary from the pickle file
# with open('t_new.pkl', 'rb') as f:
#     data = pickle.load(f)

dataset = sys.argv[1]  # e.g., "t", "z", "m", or "h"
top_k = int(sys.argv[2])  # e.g., "100", "500", "1000", or "all"
para = float(sys.argv[3])


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


# Baseline F1 from the run at para = 0.0 (given)
if dataset == "t":
    baseline_f1 = 0.8530500705218619
elif dataset == "z":
    baseline_f1 = 0.859149111937216
elif dataset == "m":
    baseline_f1 = 0.7285912974369163
elif dataset == "h":
    baseline_f1 = 0.7018601446779194

# Optionally, extract individual variables
sequences = data['sequences']
pME = data['pME']
pELF = data['pELF']
pIL = data['pIL']
pEE = data['pEE']
pELM = data['pELM']
pEO = data['pEO']
pELL = data['pELL']
emissions5 = data['emissions5']
emissions3 = data['emissions3']
lengths = data['lengths']
trueSeqs = data['trueSeqs']
testGenes = data['testGenes']
B3 = data['B3']
B5 = data['B5']


fbp5 = emissions5.copy()
fbp3 = emissions3.copy()

fbp5[:] = -np.inf
fbp3[:] = -np.inf



# ############ ############ ############ ############ ############ ############ ############ 

# 保存原始標準輸出，方便後續還原
original_stdout = sys.stdout


if dataset == "t":
    len_index = 1117
elif dataset == "z":
    len_index = 825
elif dataset == "m":
    len_index = 1212
elif dataset == "h":
    len_index = 1629
    



for para in [para]:
    # 針對所有基因進行處理：從 index 0 到 1117
    for i in range(0, len_index):
        # original_stdout.write(f"Running index {i}...\n")
        
        if dataset == "t":
            input_filename = f"./0_t_result/t_result_{top_k}/000_arabidopsis_g_{i}.txt"  # 用 i 作為文件名的一部分，避免重複寫入同一文件
        elif dataset == "z":
            input_filename = f"./0_z_result/z_result_{top_k}/000_zebrafish_g_{i}.txt"
        elif dataset == "m":
            input_filename = f"./0_m_result/m_result_{top_k}/000_mouse_g_{i}.txt"
        elif dataset == "h":
            input_filename = f"./0_h_result/h_result_{top_k}/000_human_g_{i}.txt"

        sys.stdout = original_stdout

        # original_stdout.write("Running viterbi...\n")
        # input_filename = f"./t_result_100/000_arabidopsis_g_{i}.txt"
        with open(input_filename, "r") as f:
            text = f.read()

        # 定義解析已標注的剪接位點的正則表達式
        pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
        pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
        # 定義解析 SMsplice 剪接位點的正則表達式
        pattern_smsplice_5ss = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
        pattern_smsplice_3ss = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")

        def parse_list(regex, text):
            match = regex.search(text)
            if not match:
                return set()
            inside = match.group(1).strip()
            if not inside:
                return set()
            items = re.split(r"[\s,]+", inside)
            return set(map(int, items))
        
        annotated_5prime = parse_list(pattern_5ss, text)
        annotated_3prime = parse_list(pattern_3ss, text)
        viterbi_5prime = parse_list(pattern_smsplice_5ss, text)
        viterbi_3prime = parse_list(pattern_smsplice_3ss, text)

        # 定義解析排序後預測結果的正則表達式（用於計算正確率）
        pattern_5prime_block = re.compile(
            r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
            re.DOTALL
        )
        pattern_3prime_block = re.compile(
            r"Sorted 3['′] Splice Sites .*?\n(.*)",
            re.DOTALL
        )
        pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

        def parse_predictions(pattern, text):
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

        fiveprime_preds = parse_predictions(pattern_5prime_block, text)
        threeprime_preds = parse_predictions(pattern_3prime_block, text)


        import math
        # 使用解析出的預測結果更新 fbp5 和 fbp3 中對應位置的值
        for pos, prob in fiveprime_preds:
            fbp5[i][pos] = prob

            if prob < para:
                emissions5[i][pos] = -np.inf

        for pos, prob in threeprime_preds:
            fbp3[i][pos] = prob

            if prob < para:
                emissions3[i][pos] = -np.inf


    # set_trace()





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

    f1_score = True

    predFives_all = []
    predThrees_all = []
    trueFives_all = []
    trueThrees_all = []


    if (f1_score):
        # Get the Sensitivity and Precision
        num_truePositives = 0
        num_falsePositives = 0
        num_falseNegatives = 0

        # 初始化全域最小值為正無限大
        global_min_emissions5 = np.inf
        global_min_emissions3 = np.inf


        for g, gene in enumerate(testGenes):
            L = lengths[g]
            predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
            trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

            predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
            trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
            
            # if args.print_predictions: 
            # print(gene)
            # print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)   
            # print("\tAnnotated Threes:", trueThrees, "Predicted Threes:", predThrees)
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
        print(f"prob = {para}", "Recall", ssSens, "Precision", ssPrec, "f1", f1) 
        
        # print("prob = 0.0 Recall 0.8455959454736106 Precision 0.8606367840626111 f1 0.8530500705218619")
        print(f"f1 add {100*(f1 - baseline_f1)}%")
        

