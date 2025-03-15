import numpy as np
import pandas as pd
import time, argparse
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
import pickle
from SMsplice import *
from ipdb import set_trace
import sys

my_seed = int(sys.argv[1])  # 读取命令行参数并转换为整数
top_k = int(sys.argv[2])

# my_seed = 2
np.random.seed(my_seed)
print(f"seed = {my_seed}")

if len(sys.argv) < 3:
    print("Usage: python runSMsplice_fba.py <index> <top_k>")
    sys.exit(1)

checkpoint_interval = 1

# 加载 pickle 文件中的字典
with open(f'm_new_{my_seed}.pkl', 'rb') as f:
    data = pickle.load(f)

# 提取各个变量
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

with open(f'm_pred_all_new_{my_seed}.pkl', 'rb') as f:
    data_pred_all = pickle.load(f)

pred_all = data_pred_all['pred_all']
loglik = pred_all[1]

with open(f'm_predictions_new_{my_seed}.pkl', 'rb') as f:
    data = pickle.load(f)

# 保存原始的 sys.stdout
original_stdout = sys.stdout

# 1212
for i in range(80, 1212):  # 循环遍历指定索引范围
    # 打印当前正在处理的索引，输出到原始标准输出（终端）
    original_stdout.write(f"Running index {i}...\n")
    
    fold_name = f"./{my_seed}_m_result/m_result_{top_k}"
    output_filename = f"./{my_seed}_m_result/m_result_{top_k}/000_mouse_g_{i}.txt"  # 注意这里用 i 作为文件名的一部分，避免重复写入同一个文件
    import os
    os.makedirs(fold_name, exist_ok=True)

    # 重定向输出到文件
    sys.stdout = open(output_filename, "w")
    
    print(f"Gene = {testGenes[i]}")
    print(f"index = {i}")

    trueFives_all = data['trueFives_all']
    trueThrees_all = data['trueThrees_all']

    print(f"Annotated 5SS: {trueFives_all[i]}")
    print(f"Annotated 3SS: {trueThrees_all[i]}")

    predFives_all = data['predFives_all']
    predThrees_all = data['predThrees_all']

    print(f"SMsplice 5SS: {predFives_all[i]}")
    print(f"SMsplice 3SS: {predThrees_all[i]}")

    print("top_k = ", top_k)
    posterior, logZ = forward_backward_low_memory(sequences[i], pME, pELF, pIL, pEE, pELM, pEO, pELL,
                                                   emissions5[i], emissions3[i], lengths[i],
                                                   checkpoint_interval, top_k)

    print("Partition function (log Z):", logZ)

    five_positions = []  # 存放有 5' 剪接点的 (位置, 概率)
    three_positions = []  # 存放有 3' 剪接点的 (位置, 概率)

    for pos in range(1, lengths[i]):
        if 5 in posterior[pos]:
            five_positions.append((pos-1, posterior[pos][5]))
        if 3 in posterior[pos]:
            three_positions.append((pos-1, posterior[pos][3]))

    # 按概率从高到低排序
    five_positions.sort(key=lambda x: x[1], reverse=True)
    three_positions.sort(key=lambda x: x[1], reverse=True)

    print("\nSorted 5' Splice Sites (High to Low Probability):")
    for pos, prob in five_positions:
        print(f"Position {pos}: {prob}")

    print("\nSorted 3' Splice Sites (High to Low Probability):")
    for pos, prob in three_positions:
        print(f"Position {pos}: {prob}")

    # 关闭文件并恢复原始的 sys.stdout
    sys.stdout.close()
    sys.stdout = original_stdout
