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
# tok_k = sys.argv[2]

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
    对所有文件整体计算 Viterbi 预测指标：
      对每个文件分别计算真实（Annotated）剪接位点数、Viterbi 预测数以及预测正确数（TP），然后累加后计算整体指标。
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
      对于每个文件，选择那些概率>=threshold 的预测（不论是否在 SMsplice 内），
      然后累加各文件的真实数、预测数和真阳性数后计算整体指标。
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, _, _ = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        preds = list(df[df["prob"] >= threshold]["position"].tolist())
        total_pred += len(preds)
        tp = len(set(preds).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

def plot_threshold_metrics(dataset, topk_list, thresholds):
    """
    对给定的多个 top_k 值绘制指标图：
      - 每个 top_k 值绘制一个子图，显示基于阈值的指标（Recall、Precision、F1）随阈值变化的曲线。
      - 同时添加 Viterbi 指标作为水平虚线。
      - 所有子图使用相同的 y 轴范围，并将图例整体放置在最左边子图外侧。
    """
    fig, axes = plt.subplots(1, len(topk_list), figsize=(15, 5))
    if len(topk_list) == 1:
        axes = [axes]
    
    # 用于保存所有计算结果，稍后写入 txt 文件
    all_metrics = {}
    
    for ax, top_k_val in zip(axes, topk_list):
        files = get_files_for_topk(dataset, top_k_val)
        if not files:
            ax.set_title(f"Top_k = {top_k_val} (No files)")
            continue
        
        # Compute Viterbi metrics and plot as horizontal dotted lines
        v_recall, v_precision, v_f1 = evaluate_all_files_viterbi(files)
        ax.axhline(y=v_recall, color='blue', linestyle=':', label="Viterbi Recall")
        ax.axhline(y=v_precision, color='green', linestyle=':', label="Viterbi Precision")
        ax.axhline(y=v_f1, color='red', linestyle=':', label="Viterbi F1")
        
        recalls = []
        precisions = []
        f1s = []
        threshold_metrics = {}  # store metrics for each threshold for current top_k
        
        for thr in thresholds:
            met = evaluate_all_files_threshold(files, thr)
            recalls.append(met[0])
            precisions.append(met[1])
            f1s.append(met[2])
            threshold_metrics[thr] = {
                "Recall": met[0],
                "Precision": met[1],
                "F1": met[2]
            }
        
        # Save Viterbi and threshold metrics for this top_k value
        all_metrics[top_k_val] = {
            "Viterbi": {
                "Recall": v_recall,
                "Precision": v_precision,
                "F1": v_f1
            },
            "Thresholds": threshold_metrics
        }
        
        ax.plot(thresholds, recalls, marker='o', color='blue', label="Threshold Recall")
        ax.plot(thresholds, precisions, marker='s', color='green', label="Threshold Precision")
        ax.plot(thresholds, f1s, marker='^', color='red', label="Threshold F1")
        ax.set_xlabel("≥ FBP", fontsize=12)
        if dataset == "t":
            ax.set_title(f"Arabidopsis (Top k = {top_k_val})", fontsize=14)
        elif dataset == "z":
            ax.set_title(f"Zebrafish (Top k = {top_k_val})", fontsize=14)
        elif dataset == "m":
            ax.set_title(f"Mouse (Top k = {top_k_val})", fontsize=14)
        elif dataset == "h":
            ax.set_title(f"Human (Top k = {top_k_val})", fontsize=14)
        else:
            ax.set_title(f"Dataset (Top k = {top_k_val})", fontsize=14)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True)
        ax.invert_xaxis()  # 反转 x 轴，使较高的阈值在左侧

    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0.15, 0, 1, 1])
    fig.legend(
        handles, 
        labels, 
        loc='upper left', 
        bbox_to_anchor=(0.0, 1.0),
        fontsize=12
    )
    
    # 统一设置所有子图的 y 轴范围
    global_min = 1.0
    global_max = 0.0
    for ax in axes:
        y0, y1 = ax.get_ylim()
        global_min = min(global_min, y0)
        global_max = max(global_max, y1)
    for ax in axes:
        ax.set_ylim(global_min, global_max)
    
    # Save the figure
    outpng = f"./0_{dataset}_result/f1_V_only_FBP.png"
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    plt.savefig(outpng, dpi=150, bbox_inches="tight")
    print(f"Threshold metrics figure with Viterbi lines saved to {outpng}")
    plt.close()
    
    # Save metrics values to a text file
    outtxt = f"./0_{dataset}_result/f1_V_only_FBP.txt"
    with open(outtxt, "w") as f:
        for topk, metrics in all_metrics.items():
            f.write(f"Top k = {topk}\n")
            f.write("  Viterbi Metrics:\n")
            f.write(f"    Recall   : {metrics['Viterbi']['Recall']:.4f}\n")
            f.write(f"    Precision: {metrics['Viterbi']['Precision']:.4f}\n")
            f.write(f"    F1       : {metrics['Viterbi']['F1']:.4f}\n")
            f.write("  Threshold Metrics:\n")
            for thr, m in metrics["Thresholds"].items():
                f.write(f"    Threshold {thr:.2f} -> Recall: {m['Recall']:.4f}, Precision: {m['Precision']:.4f}, F1: {m['F1']:.4f}\n")
            f.write("\n")
    print(f"Metrics values saved to {outtxt}")

def main():
    topk_list = [100, 500, 1000]
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
    plot_threshold_metrics(dataset, topk_list, thresholds)
        
if __name__ == "__main__":
    main()
