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
      - annotated_5: Annotated 5' 剪接位点（集合，GT）
      - annotated_3: Annotated 3' 剪接位点（集合，GT）
      - viterbi_5: SMsplice 5' 预测（集合）
      - viterbi_3: SMsplice 3' 预测（集合）
    """
    with open(filename, "r") as f:
        text = f.read()
    
    # 解析 Annotated 位点（真实剪接位点 GT）
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    # 解析 SMsplice（Viterbi预测）位点
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
    根据累计的真阳性、预测总数和真实总数计算 Recall, Precision, F1
    """
    recall = tp / total_gt if total_gt > 0 else 0
    precision = tp / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1

# --------------------------------------------------------------------
# Line one: All Viterbi (no FBP threshold)
# 使用 viterbi_5 ∪ viterbi_3（不做概率过滤）
def evaluate_all_files_all_viterbi(files):
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        # Retrieve GT and Viterbi sets directly.
        _, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        pred_all_v = list(viterbi_5.union(viterbi_3))
        total_pred += len(pred_all_v)
        tp = len(set(pred_all_v).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)


# --------------------------------------------------------------------
# Line two: Viterbi with >= FBP
# 从 viterbi_5 ∪ viterbi_3 中取出各位置的概率（若未在 DataFrame 中出现则视为 0），
# 并移除概率 < threshold 的预测。
def evaluate_all_files_viterbi_filtered(files, threshold):
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        filtered_v5 = []
        filtered_v3 = []
        # 对5'预测单独过滤
        for pos in viterbi_5:
            rows = df[(df["position"] == pos) & (df["type"] == "5prime")]
            if not rows.empty:
                max_prob = rows["prob"].iloc[0]
            else:
                max_prob = 0
            if max_prob >= threshold:
                filtered_v5.append(pos)
        # 对3'预测单独过滤
        for pos in viterbi_3:
            rows = df[(df["position"] == pos) & (df["type"] == "3prime")]
            if not rows.empty:
                max_prob = rows["prob"].iloc[0]
            else:
                max_prob = 0
            if max_prob >= threshold:
                filtered_v3.append(pos)
        # 取两部分的并集
        filtered_v = list(set(filtered_v5).union(set(filtered_v3)))
        total_pred += len(filtered_v)
        tp = len(set(filtered_v).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)


def evaluate_all_files_viterbi_plus_non(files, threshold):
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
        additional = list(df[(~df["in_viterbi"]) & (df["prob"] >= threshold)]["position"].tolist())
        pred_c = list(set(pred_v).union(set(additional)))
        total_pred += len(pred_c)
        tp = len(set(pred_c).intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

# --------------------------------------------------------------------
# # Line three: Viterbi + Non-Viterbi (>= FBP)
# # 使用所有 Viterbi 预测（viterbi_5 ∪ viterbi_3，不进行阈值过滤）
# # 与非 Viterbi 中概率 >= threshold 的预测取并集。
# def evaluate_all_files_viterbi_plus_non(files, threshold):
#     total_gt = 0
#     total_pred = 0
#     total_tp = 0
#     for fname in files:
#         df, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
#         gt = list(annotated_5) + list(annotated_3)
#         total_gt += len(gt)
#         union_v = viterbi_5.union(viterbi_3)
#         non_v = set(df[(~df["in_viterbi"]) & (df["prob"] >= threshold)]["position"].tolist())
#         combined = union_v.union(non_v)
#         total_pred += len(combined)
#         tp = len(combined.intersection(set(gt)))
#         total_tp += tp
#     return calc_metrics_from_counts(total_tp, total_pred, total_gt)



# --------------------------------------------------------------------
# Line four: All predictions with >= FBP
# 直接使用所有预测中概率 >= threshold 的记录（不区分 SMsplice 标记）。
def evaluate_all_files_combined(files, threshold):
    total_gt = 0
    total_pred = 0
    total_tp = 0
    for fname in files:
        df, annotated_5, annotated_3, _, _ = parse_splice_file(fname)
        gt = list(annotated_5) + list(annotated_3)
        total_gt += len(gt)
        preds = set(df[df["prob"] >= threshold]["position"].tolist())
        total_pred += len(preds)
        tp = len(preds.intersection(set(gt)))
        total_tp += tp
    return calc_metrics_from_counts(total_tp, total_pred, total_gt)

# --------------------------------------------------------------------
def plot_combined_metrics(dataset, topk_list, thresholds):
    """
    绘制指标图：
      - x 轴为不同的 FBP 阈值 (≥ FBP)
      - 对每个 top_k，绘制四条评价曲线：
          Line 1: All Viterbi (viterbi_5 ∪ viterbi_3，无概率过滤)
          Line 2: Viterbi (>= FBP)（从 viterbi_5 ∪ viterbi_3 中筛选出概率 >= FBP 的记录）
          Line 3: Viterbi+Non-Viterbi (>= FBP)
          Line 4: All (>= FBP)
      - Recall 使用蓝色，Precision 使用绿色，F1 使用红色；
        不同方法采用不同线型：
            * Line 1: 实线 ("-")
            * Line 2: 虚线 ("--")
            * Line 3: 点划线 ("-.")
            * Line 4: 点线 (":")
    """
    fig, axes = plt.subplots(1, len(topk_list), figsize=(14, 6))
    if len(topk_list) == 1:
        axes = [axes]
    
    for ax, top_k_val in zip(axes, topk_list):
        files = get_files_for_topk(dataset, top_k_val)
        if not files:
            ax.set_title(f"Top_k = {top_k_val} (No files)")
            continue
        
        # Line 1: All Viterbi (no threshold)
        all_v_metrics = evaluate_all_files_all_viterbi(files)
        line1_recall = all_v_metrics[0]
        line1_precision = all_v_metrics[1]
        line1_f1 = all_v_metrics[2]
        line1_recalls = [line1_recall] * len(thresholds)
        line1_precisions = [line1_precision] * len(thresholds)
        line1_f1s = [line1_f1] * len(thresholds)
        
        # Lines 2, 3, 4 computed per threshold
        line2_recalls, line2_precisions, line2_f1s = [], [], []
        line3_recalls, line3_precisions, line3_f1s = [], [], []
        line4_recalls, line4_precisions, line4_f1s = [], [], []
        for thr in thresholds:
            met2 = evaluate_all_files_viterbi_filtered(files, thr)
            met3 = evaluate_all_files_viterbi_plus_non(files, thr)
            met4 = evaluate_all_files_combined(files, thr)
            line2_recalls.append(met2[0])
            line2_precisions.append(met2[1])
            line2_f1s.append(met2[2])
            line3_recalls.append(met3[0])
            line3_precisions.append(met3[1])
            line3_f1s.append(met3[2])
            line4_recalls.append(met4[0])
            line4_precisions.append(met4[1])
            line4_f1s.append(met4[2])
        
        # 绘制 Recall 曲线（蓝色）
        ax.plot(thresholds, line1_recalls, marker='o', linestyle='-', color='blue', label="All Viterbi")
        ax.plot(thresholds, line2_recalls, marker='o', linestyle='--', color='blue', label="Viterbi (>= FBP)")
        ax.plot(thresholds, line3_recalls, marker='o', linestyle='-.', color='blue', label="Viterbi+Non-Viterbi (>= FBP)")
        ax.plot(thresholds, line4_recalls, marker='o', linestyle=':', color='blue', label="All (>= FBP)")
        
        # 绘制 Precision 曲线（绿色）
        ax.plot(thresholds, line1_precisions, marker='s', linestyle='-', color='green', label="All Viterbi")
        ax.plot(thresholds, line2_precisions, marker='s', linestyle='--', color='green', label="Viterbi (>= FBP)")
        ax.plot(thresholds, line3_precisions, marker='s', linestyle='-.', color='green', label="Viterbi+Non-Viterbi (>= FBP)")
        ax.plot(thresholds, line4_precisions, marker='s', linestyle=':', color='green', label="All (>= FBP)")
        
        # 绘制 F1 曲线（红色）
        ax.plot(thresholds, line1_f1s, marker='^', linestyle='-', color='red', label="All Viterbi")
        ax.plot(thresholds, line2_f1s, marker='^', linestyle='--', color='red', label="Viterbi (>= FBP)")
        ax.plot(thresholds, line3_f1s, marker='^', linestyle='-.', color='red', label="Viterbi+Non-Viterbi (>= FBP)")
        ax.plot(thresholds, line4_f1s, marker='^', linestyle=':', color='red', label="All (>= FBP)")
        
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
    
    # 创建统一图例（去重标签）
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    fig.legend(list(unique.values()), list(unique.keys()), loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=10)
    
    # 统一设置所有子图的 y 轴范围
    global_min, global_max = 1.0, 0.0
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
    将每个 top_k 下不同阈值（≥ FBP）的四种预测方式指标写入文本文件：
      1. All Viterbi (no FBP)
      2. Viterbi (>= FBP)
      3. Viterbi + Non-Viterbi (>= FBP)
      4. All (>= FBP)
    同时，增加各方法 F1 提升值，相对于 All Viterbi (no FBP) 基线。
    """
    outtxt = f"./0_{dataset}_result/f1_V_with_FBP_{top_k}.txt"
    os.makedirs(os.path.dirname(outtxt), exist_ok=True)
    with open(outtxt, "w") as f:
        for top_k_val in topk_list:
            files = get_files_for_topk(dataset, top_k_val)
            if not files:
                f.write(f"Top_k = {top_k_val}: No files found\n")
                continue
            f.write(f"Top_k = {top_k_val} Metrics:\n")
            # 计算基线：All Viterbi (no FBP)
            met_line1 = evaluate_all_files_all_viterbi(files)
            f.write(f"  All Viterbi (no FBP): Recall: {met_line1[0]:.8f}, Precision: {met_line1[1]:.8f}, F1: {met_line1[2]:.8f}\n")
            for thr in thresholds:
                met_line2 = evaluate_all_files_viterbi_filtered(files, thr)
                met_line3 = evaluate_all_files_viterbi_plus_non(files, thr)
                met_line4 = evaluate_all_files_combined(files, thr)
                # 计算 F1 改进：各方法 F1 与基线 F1 的差值
                improve_line2 = met_line2[2] - met_line1[2]
                improve_line3 = met_line3[2] - met_line1[2]
                improve_line4 = met_line4[2] - met_line1[2]

                max_improve = max(improve_line2, improve_line3, improve_line4)
                f.write(f"  Threshold >= {thr:.2f}:\n")
                f.write(f"    >= FBP's Viterbi              -> Recall: {met_line2[0]:.8f}, Precision: {met_line2[1]:.8f}, F1: {met_line2[2]:.8f}, F1 Improvement: {100*improve_line2:+.8f}%\n")
                f.write(f"    Viterbi+Non-Viterbi (>= FBP)  -> Recall: {met_line3[0]:.8f}, Precision: {met_line3[1]:.8f}, F1: {met_line3[2]:.8f}, F1 Improvement: {100*improve_line3:+.8f}%\n")
                f.write(f"    All (>= FBP)                  -> Recall: {met_line4[0]:.8f}, Precision: {met_line4[1]:.8f}, F1: {met_line4[2]:.8f}, F1 Improvement: {100*improve_line4:+.8f}%\n")
                
                # 输出最大 F1 改进值
                f.write(f"    Max F1 Improvement: {100*max_improve:+.8f}%\n")
                print(f"    Max F1 Improvement: {100*max_improve:+.8f}%\n")

                
                
                # print(f"  Threshold >= {thr:.2f}:\n")
                # print(f"    >= FBP's Viterbi              -> Recall: {met_line2[0]:.8f}, Precision: {met_line2[1]:.8f}, F1: {met_line2[2]:.8f}, F1 Improvement: {100*improve_line2:+.8f}%\n")
                # print(f"    Viterbi+Non-Viterbi (>= FBP)  -> Recall: {met_line3[0]:.8f}, Precision: {met_line3[1]:.8f}, F1: {met_line3[2]:.8f}, F1 Improvement: {100*improve_line3:+.8f}%\n")
                # print(f"    All (>= FBP)                  -> Recall: {met_line4[0]:.8f}, Precision: {met_line4[1]:.8f}, F1: {met_line4[2]:.8f}, F1 Improvement: {100*improve_line4:+.8f}%\n")
            f.write("\n")
    print(f"Metrics saved to {outtxt}")



# def save_results_to_txt(dataset, topk_list, thresholds):
#     """
#     将每个 top_k 下不同阈值（≥ FBP）的四种预测方式指标写入文本文件：
#       1. All Viterbi (no FBP)
#       2. Viterbi (>= FBP)
#       3. Viterbi + Non-Viterbi (>= FBP)
#       4. All (>= FBP)
#     """
#     outtxt = f"./0_{dataset}_result/f1_V_with_FBP.txt"
#     os.makedirs(os.path.dirname(outtxt), exist_ok=True)
#     with open(outtxt, "w") as f:
#         for top_k_val in topk_list:
#             files = get_files_for_topk(dataset, top_k_val)
#             if not files:
#                 f.write(f"Top_k = {top_k_val}: No files found\n")
#                 continue
#             f.write(f"Top_k = {top_k_val} Metrics:\n")
#             met_line1 = evaluate_all_files_all_viterbi(files)
#             f.write(f"  All Viterbi (no FBP): Recall: {met_line1[0]:.8f}, Precision: {met_line1[1]:.8f}, F1: {met_line1[2]:.8f}\n")
#             for thr in thresholds:
#                 met_line2 = evaluate_all_files_viterbi_filtered(files, thr)
#                 met_line3 = evaluate_all_files_viterbi_plus_non(files, thr)
#                 met_line4 = evaluate_all_files_combined(files, thr)
#                 f.write(f"  Threshold >= {thr:.2f}:\n")
#                 f.write(f"    Viterbi (>= FBP)            -> Recall: {met_line2[0]:.8f}, Precision: {met_line2[1]:.8f}, F1: {met_line2[2]:.8f}\n")
#                 f.write(f"    Viterbi+Non-Viterbi (>= FBP)  -> Recall: {met_line3[0]:.8f}, Precision: {met_line3[1]:.8f}, F1: {met_line3[2]:.8f}\n")
#                 f.write(f"    All (>= FBP)                -> Recall: {met_line4[0]:.8f}, Precision: {met_line4[1]:.8f}, F1: {met_line4[2]:.8f}\n")
#             f.write("\n")
#     print(f"Metrics saved to {outtxt}")

def main():
    topk_list = [top_k]  # 可根据需要调整多个 top_k 值
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
    plot_combined_metrics(dataset, topk_list, thresholds)
    save_results_to_txt(dataset, topk_list, thresholds)
        
if __name__ == "__main__":
    main()
