#!/usr/bin/env python3
import sys
import re
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# dataset 参数：例如 "t", "z", "m", "h"
dataset = sys.argv[1]

def parse_splice_file(filename):
    """
    解析一个文本文件，返回：
      - df: 包含排序预测的 DataFrame（每行记录位置、概率、类型、是否正确、是否属于 SMsplice/Viterbi）
      - annotated_5: Annotated 5' 剪接位点（集合）
      - annotated_3: Annotated 3' 剪接位点（集合）
      - viterbi_5: SMsplice 5' 预测（集合）
      - viterbi_3: SMsplice 3' 预测（集合）
    """
    with open(filename, "r") as f:
        text = f.read()
    
    # 解析 Annotated 位点
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    # 解析 SMsplice（Viterbi）预测
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
    
    annotated_5 = parse_list(pattern_5ss)
    annotated_3 = parse_list(pattern_3ss)
    viterbi_5 = parse_list(pattern_smsplice_5ss)
    viterbi_3 = parse_list(pattern_smsplice_3ss)
    
    # 解析排序预测部分（5' 与 3' 分开）
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
    # 5' 预测
    for (pos, prob) in fiveprime_preds:
        is_correct = (pos in annotated_5)
        in_viterbi = (pos in viterbi_5)
        rows.append((pos, prob, "5prime", is_correct, in_viterbi))
    # 3' 预测
    for (pos, prob) in threeprime_preds:
        is_correct = (pos in annotated_3)
        in_viterbi = (pos in viterbi_3)
        rows.append((pos, prob, "3prime", is_correct, in_viterbi))
    
    df = pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "in_viterbi"])
    return df, annotated_5, annotated_3, viterbi_5, viterbi_3

def get_files_for_topk(dataset, top_k_val):
    """
    根据 dataset 和 top_k 值，返回匹配的文件列表
    """
    if dataset == "t":
        pattern = f"./0_t_result/t_result_{top_k_val}/000_arabidopsis_g_*.txt"
    elif dataset == "z":
        pattern = f"./0_z_result/z_result_{top_k_val}/000_zebrafish_g_*.txt"
    elif dataset == "m":
        pattern = f"./0_m_result/m_result_{top_k_val}/000_mouse_g_*.txt"
    elif dataset == "h":
        pattern = f"./0_h_result/h_result_{top_k_val}/000_human_g_*.txt"
    else:
        print("未知 dataset")
        return []
    return sorted(glob.glob(pattern))

def calc_metrics_from_counts(tp, total_pred, total_gt):
    """
    根据累计的真阳性、预测总数和真实总数计算 recall, precision, F1
    """
    recall = tp / total_gt if total_gt > 0 else 0
    precision = tp / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1

def evaluate_all_files_viterbi(files):
    """
    对所有文件整体计算 Viterbi 预测指标（直接取文本中 SMsplice 的结果）。
    对每个文件分别计算真实（Annotated）剪接位点数、预测数以及预测正确数（TP），然后累加。
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        pred_v = list(viterbi_5) + list(viterbi_3)
        total_pred += len(pred_v)
        tp = len(set(pred_v).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

def evaluate_all_files_threshold(files, threshold):
    """
    对所有文件整体计算 Combined 预测指标：
      对于每个文件，在 Viterbi 预测的基础上，加入那些不在 SMsplice 内且概率>=threshold 的预测，
      然后累加各文件的真实数、预测数和真阳性数后计算整体指标。
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        pred_v = list(viterbi_5) + list(viterbi_3)
        # 加入阈值判定：只添加不在 viterbi 内且 prob >= threshold 的位点
        additional = list(df[(~df["in_viterbi"]) & (df["prob"] >= threshold)]["position"].tolist())
        pred_c = list(set(pred_v).union(set(additional)))
        total_pred += len(pred_c)
        tp = len(set(pred_c).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

def save_results_to_txt(dataset, topk_list, thresholds,
                        v_recall_dict, v_precision_dict, v_f1_dict,
                        recall_dict, precision_dict, f1_dict):
    """
    将每个 top_k 下的 Viterbi 基线以及各阈值（V+FBP）指标写入文本文件
    """
    outtxt = f"./0_{dataset}_result/different_top_k_line_three.txt"
    os.makedirs(os.path.dirname(outtxt), exist_ok=True)
    with open(outtxt, "w") as f:
        for top_k_val in topk_list:
            if top_k_val not in v_recall_dict:
                f.write(f"Top_k = {top_k_val}: 未找到对应文件\n\n")
                continue
            f.write(f"Top_k = {top_k_val} Viterbi Baseline:\n")
            f.write(f"  Recall: {v_recall_dict[top_k_val]:.4f}\n")
            f.write(f"  Precision: {v_precision_dict[top_k_val]:.4f}\n")
            f.write(f"  F1: {v_f1_dict[top_k_val]:.4f}\n")
            f.write("\nThreshold Metrics (V+FBP):\n")
            for thr, rec, prec, f1 in zip(thresholds, recall_dict[top_k_val],
                                          precision_dict[top_k_val], f1_dict[top_k_val]):
                f.write(f"  Threshold >= {thr:.2f} : Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}\n")
            f.write("\n")
    print(f"评估结果已保存至 {outtxt}")

def main():
    # 需要比较的 top_k 列表
    topk_list = [100, 500, 1000]
    # 定义阈值列表，从 0.95 到 0.40
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    
    # 为不同 top_k 设置不同的 marker 形状
    marker_map = {100: 'o', 500: 's', 1000: '^'}
    # 为不同 top_k 指定固定的颜色
    color_map = {100: 'blue', 500: 'green', 1000: 'red'}

    # 用字典存储不同 top_k 对应的评估结果
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}
    # 同时存储各 top_k 的 Viterbi 基线
    v_recall_dict = {}
    v_precision_dict = {}
    v_f1_dict = {}

    for top_k_val in topk_list:
        files = get_files_for_topk(dataset, top_k_val)
        if not files:
            print(f"没有找到 top_k = {top_k_val} 的文件")
            continue

        # 1) 计算 Viterbi 基线
        v_recall, v_precision, v_f1 = evaluate_all_files_viterbi(files)
        v_recall_dict[top_k_val] = v_recall
        v_precision_dict[top_k_val] = v_precision
        v_f1_dict[top_k_val] = v_f1

        # 2) 对每个阈值计算 (V+FBP) 的 Recall, Precision, F1
        thr_recalls = []
        thr_precisions = []
        thr_f1s = []
        for thr in thresholds:
            r, p, f = evaluate_all_files_threshold(files, thr)
            thr_recalls.append(r)
            thr_precisions.append(p)
            thr_f1s.append(f)

        recall_dict[top_k_val] = thr_recalls
        precision_dict[top_k_val] = thr_precisions
        f1_dict[top_k_val] = thr_f1s

        print(f"Top_k = {top_k_val} 的 Viterbi 基线： Recall={v_recall:.4f}, Precision={v_precision:.4f}, F1={v_f1:.4f}")

    # -------------------------
    # 设置全局样式：使用默认样式，并手动设置背景和网格
    plt.style.use('default')
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': 'lightgray',
        'grid.linestyle': '--'
    })

    # -------------------------
    # 绘图：绘制 Recall / Precision / F1 三张图
    # -------------------------

    # 1) Recall 图
    plt.figure(figsize=(8, 6))
    for top_k_val in topk_list:
        if top_k_val in recall_dict:
            plt.plot(thresholds, recall_dict[top_k_val],
                     marker=marker_map[top_k_val], markersize=8,
                     linewidth=2, color=color_map[top_k_val],
                     label=f"V+FBP Recall (k={top_k_val})")
            plt.axhline(y=v_recall_dict[top_k_val], linestyle='--', linewidth=1.5,
                        color=color_map[top_k_val],
                        label=f"V Recall (k={top_k_val})")
    plt.xlabel("≥ FBP", fontsize=14)
    plt.ylabel("Recall", fontsize=14)
    if dataset == "t":
        plt.title("arabidopsis - Recall vs. Threshold", fontsize=16)
    elif dataset == "z":
        plt.title("zebrafish - Recall vs. Threshold", fontsize=16)
    elif dataset == "m":
        plt.title("mouse - Recall vs. Threshold", fontsize=16)
    elif dataset == "h":
        plt.title("human - Recall vs. Threshold", fontsize=16)
    else:
        plt.title("Recall vs. Threshold", fontsize=16)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    outpng_recall = f"./0_{dataset}_result/different_top_k_line_three_recall.png"
    os.makedirs(os.path.dirname(outpng_recall), exist_ok=True)
    plt.savefig(outpng_recall, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Recall 图已保存至 {outpng_recall}")

    # 2) Precision 图
    plt.figure(figsize=(8, 6))
    for top_k_val in topk_list:
        if top_k_val in precision_dict:
            plt.plot(thresholds, precision_dict[top_k_val],
                     marker=marker_map[top_k_val], markersize=8,
                     linewidth=2, color=color_map[top_k_val],
                     label=f"V+FBP Precision (k={top_k_val})")
            plt.axhline(y=v_precision_dict[top_k_val], linestyle='--', linewidth=1.5,
                        color=color_map[top_k_val],
                        label=f"V Precision (k={top_k_val})")
    plt.xlabel("≥ FBP", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    if dataset == "t":
        plt.title("arabidopsis - Precision vs. Threshold", fontsize=16)
    elif dataset == "z":
        plt.title("zebrafish - Precision vs. Threshold", fontsize=16)
    elif dataset == "m":
        plt.title("mouse - Precision vs. Threshold", fontsize=16)
    elif dataset == "h":
        plt.title("human - Precision vs. Threshold", fontsize=16)
    else:
        plt.title("Precision vs. Threshold", fontsize=16)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    outpng_precision = f"./0_{dataset}_result/different_top_k_line_three_precision.png"
    os.makedirs(os.path.dirname(outpng_precision), exist_ok=True)
    plt.savefig(outpng_precision, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Precision 图已保存至 {outpng_precision}")

    # 3) F1 图
    plt.figure(figsize=(8, 6))
    for top_k_val in topk_list:
        if top_k_val in f1_dict:
            plt.plot(thresholds, f1_dict[top_k_val],
                     marker=marker_map[top_k_val], markersize=8,
                     linewidth=2, color=color_map[top_k_val],
                     label=f"V+FBP F1 (k={top_k_val})")
            plt.axhline(y=v_f1_dict[top_k_val], linestyle='--', linewidth=1.5,
                        color=color_map[top_k_val],
                        label=f"V F1 (k={top_k_val})")
    plt.xlabel("≥ FBP", fontsize=14)
    plt.ylabel("F1-score", fontsize=14)
    if dataset == "t":
        plt.title("arabidopsis - F1 vs. Threshold", fontsize=16)
    elif dataset == "z":
        plt.title("zebrafish - F1 vs. Threshold", fontsize=16)
    elif dataset == "m":
        plt.title("mouse - F1 vs. Threshold", fontsize=16)
    elif dataset == "h":
        plt.title("human - F1 vs. Threshold", fontsize=16)
    else:
        plt.title("F1 vs. Threshold", fontsize=16)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    outpng_f1 = f"./0_{dataset}_result/different_top_k_line_three_f1.png"
    os.makedirs(os.path.dirname(outpng_f1), exist_ok=True)
    plt.savefig(outpng_f1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"F1 图已保存至 {outpng_f1}")

    # 将结果数据写入文本文件
    save_results_to_txt(dataset, topk_list, thresholds,
                        v_recall_dict, v_precision_dict, v_f1_dict,
                        recall_dict, precision_dict, f1_dict)

if __name__ == "__main__":
    main()
