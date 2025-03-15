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
top_k = int(sys.argv[2])

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

def evaluate_all_files_viterbi(files, threshold):
    """
    对所有文件整体计算 Viterbi 预测指标：
      仅使用 in_viterbi 为 True 且概率 >= threshold 的预测。
      对每个文件分别计算真实（Annotated）剪接位点数、预测数以及预测正确数（TP），然后累加，
      最后计算 Recall、Precision 和 F1。
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, _, _ = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        pred_v = list(df[(df["in_viterbi"]) & (df["prob"] >= threshold)]["position"].tolist())
        total_pred += len(pred_v)
        tp = len(set(pred_v).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

def evaluate_all_files_threshold(files, threshold):
    """
    对所有文件整体计算 Combined 预测指标：
      对于每个文件，先取出满足概率 >= threshold 的 Viterbi 预测，
      再加入那些不在 Viterbi 内但概率 >= threshold 的预测，
      最后计算整体的 Recall、Precision 和 F1。
      （实际上就是使用所有概率 >= threshold 的预测）
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, _, _ = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        # Combined 预测：所有概率 >= threshold 的预测（不论是否 in_viterbi）
        pred_c = list(df[df["prob"] >= threshold]["position"].tolist())
        total_pred += len(pred_c)
        tp = len(set(pred_c).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

def plot_combined_metrics(dataset, topk_list, thresholds):
    """
    对给定的多个 top_k 值绘制指标图：
      - 每个 top_k 值绘制一个子图，显示 Viterbi 指标（仅使用 in_viterbi 且概率 >= threshold 的预测）
        与 Combined 指标（所有预测概率 >= threshold）的 Recall、Precision、F1 随阈值变化的曲线。
      - 所有子图使用相同的 y 轴范围，并将图例放在最左边子图的外侧。
    """
    fig, axes = plt.subplots(1, len(topk_list), figsize=(10, 5))
    if len(topk_list) == 1:
        axes = [axes]
    
    for ax, top_k_val in zip(axes, topk_list):
        files = get_files_for_topk(dataset, top_k_val)
        if not files:
            ax.set_title(f"Top_k = {top_k_val} (No files)")
            continue
        
        viterbi_recalls = []
        viterbi_precisions = []
        viterbi_f1s = []
        combined_recalls = []
        combined_precisions = []
        combined_f1s = []
        
        for thr in thresholds:
            met_v = evaluate_all_files_viterbi(files, thr)
            met_c = evaluate_all_files_threshold(files, thr)
            viterbi_recalls.append(met_v[0])
            viterbi_precisions.append(met_v[1])
            viterbi_f1s.append(met_v[2])
            combined_recalls.append(met_c[0])
            combined_precisions.append(met_c[1])
            combined_f1s.append(met_c[2])
        
        ax.plot(thresholds, combined_recalls, marker='o', color='blue', label="Combined Recall")
        ax.plot(thresholds, combined_precisions, marker='s', color='green', label="Combined Precision")
        ax.plot(thresholds, combined_f1s, marker='^', color='red', label="Combined F1")
        ax.plot(thresholds, viterbi_recalls, marker='o', linestyle='--', color='blue', label="Viterbi Recall")
        ax.plot(thresholds, viterbi_precisions, marker='s', linestyle='--', color='green', label="Viterbi Precision")
        ax.plot(thresholds, viterbi_f1s, marker='^', linestyle='--', color='red', label="Viterbi F1")
        
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
        ax.invert_xaxis()
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0.15, 0, 1, 1])
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=12)
    
    # 统一设置所有子图的 y 轴范围
    global_min = 1.0
    global_max = 0.0
    for ax in axes:
        y0, y1 = ax.get_ylim()
        global_min = min(global_min, y0)
        global_max = max(global_max, y1)
    for ax in axes:
        ax.set_ylim(global_min, global_max)
    
    outpng = f"./0_{dataset}_result/f1_V_with_FBP.png"
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    plt.savefig(outpng, dpi=150, bbox_inches="tight")
    print(f"Combined figure saved to {outpng}")
    plt.close()

def save_results_to_txt(dataset, topk_list, thresholds):
    """
    将每个 top_k 下的各阈值（Combined 与 Viterbi）的指标写入文字文件
    """
    outtxt = f"./0_{dataset}_result/f1_V_with_FBP.txt"
    os.makedirs(os.path.dirname(outtxt), exist_ok=True)
    with open(outtxt, "w") as f:
        for top_k_val in topk_list:
            files = get_files_for_topk(dataset, top_k_val)
            if not files:
                f.write(f"Top_k = {top_k_val}: No files found\n")
                continue
            f.write(f"Top_k = {top_k_val} Metrics:\n")
            for thr in thresholds:
                viterbi_metrics = evaluate_all_files_viterbi(files, thr)
                combined_metrics = evaluate_all_files_threshold(files, thr)
                f.write(f"  Threshold >= {thr:.2f}:\n")
                f.write(f"    Viterbi -> Recall: {viterbi_metrics[0]:.8f}, Precision: {viterbi_metrics[1]:.8f}, F1: {viterbi_metrics[2]:.8f}\n")
                f.write(f"    Combined -> Recall: {combined_metrics[0]:.8f}, Precision: {combined_metrics[1]:.8f}, F1: {combined_metrics[2]:.8f}\n")
            f.write("\n")
    print(f"Metrics saved to {outtxt}")

def main():
    # 可以调整 topk_list 为多个值，例如：[100, 500, 1000]
    topk_list = [top_k]
    # 定义阈值列表，从 0.95 到 0.45（可根据需要调整）
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
    # 绘制图像
    plot_combined_metrics(dataset, topk_list, thresholds)
    # 保存结果指标到文本文件
    save_results_to_txt(dataset, topk_list, thresholds)
        
if __name__ == "__main__":
    main()
