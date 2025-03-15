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
from ipdb import set_trace
import sys

np.random.seed(0)



if len(sys.argv) < 2:
    print("Usage: python runSMsplice_fba.py <index>")
    sys.exit(1)


g_index = int(sys.argv[1])  # 讀取命令列參數並轉換為整數


top_k =  int(sys.argv[2])  

fold_name = f"./0_z_result/0_z_result_{top_k}"
output_filename = f"./0_z_result/0_z_result_{top_k}/000_zebrafish_g_{g_index}.txt"  # 設定輸出檔名
import os
os.makedirs(fold_name, exist_ok=True)

# 重新導向輸出到文件
sys.stdout = open(output_filename, "w")


# Load the dictionary from the pickle file
with open('z_new.pkl', 'rb') as f:
    data = pickle.load(f)

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

with open('z_pred_all_new.pkl', 'rb') as f:
    data_pred_all = pickle.load(f)

pred_all = data_pred_all['pred_all']
loglik = pred_all[1]
# Load the predictions from the pickle file
with open('z_predictions_new.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Gene = {testGenes[g_index]}")
print(f"index = {g_index}")

trueFives_all = data['trueFives_all']
trueThrees_all = data['trueThrees_all']

print(f"Annotated 5SS: {trueFives_all[g_index]}")
print(f"Annotated 3SS: {trueThrees_all[g_index]}")

predFives_all = data['predFives_all']
predThrees_all = data['predThrees_all']

print(f"SMsplice 5SS: {predFives_all[g_index]}")
print(f"SMsplice 3SS: {predThrees_all[g_index]}")

checkpoint_interval = 1


print("top_k = ", top_k)
posterior, logZ = forward_backward_low_memory(sequences[g_index], pME, pELF, pIL, pEE, pELM, pEO, pELL, emissions5[g_index], emissions3[g_index], lengths[g_index], checkpoint_interval, top_k)
# set_trace()

print("Partition function (log Z):", logZ)

five_positions = []  # 存放有 5' 剪接點的 (位置, 機率)
three_positions = []  # 存放有 3' 剪接點的 (位置, 機率)

for i in range(1, lengths[g_index]):
    # 如果該位置有 5'，存入列表
    if 5 in posterior[i]:
        five_positions.append((i-1, posterior[i][5]))

    # 如果該位置有 3'，存入列表
    if 3 in posterior[i]:
        three_positions.append((i-1, posterior[i][3]))

# 按照機率大小排序（由大到小）
five_positions.sort(key=lambda x: x[1], reverse=True)
three_positions.sort(key=lambda x: x[1], reverse=True)

# 印出 5' 剪接點機率排序
print("\nSorted 5' Splice Sites (High to Low Probability):")
for pos, prob in five_positions:
    print(f"Position {pos}: {prob}")

# 印出 3' 剪接點機率排序
print("\nSorted 3' Splice Sites (High to Low Probability):")
for pos, prob in three_positions:
    print(f"Position {pos}: {prob}")

#######
# for i in range(1, lengths[g_index]):
#     # 只印出 posterior[i] 中包含 5 或 3 的位置
#     if 5 in posterior[i] or 3 in posterior[i]:
#         print(f"Position {i-1}")
#         print("  Posterior:", posterior[i])

# 關閉輸出文件
sys.stdout.close()






