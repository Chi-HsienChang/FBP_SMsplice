#!/usr/bin/env python3

import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# 从命令行参数获取数据集标识和 top_k 值
# 使用方法：python3 plot_topk_hist.py t 100
dataset = sys.argv[1]  # 例如 "t", "z", "m", 或 "h"
top_k = int(sys.argv[2])

###############################################################################
# 辅助函数：根据数据集和类别设置标题
###############################################################################
def get_title(category):
    # category: "5SS", "3SS", 或 "ALL"
    if dataset == "t":
        if category == "5SS":
            return "Arabidopsis 5SS (#TestGene = 1117)"
        elif category == "3SS":
            return "Arabidopsis 3SS (#TestGene = 1117)"
        else:
            return "Arabidopsis 5SS+3SS (#TestGene = 1117)"
    elif dataset == "z":
        if category == "5SS":
            return "Zebrafish 5SS (#TestGene = 825)"
        elif category == "3SS":
            return "Zebrafish 3SS (#TestGene = 825)"
        else:
            return "Zebrafish 5SS+3SS (#TestGene = 825)"
    elif dataset == "m":
        if category == "5SS":
            return "Mouse 5SS (#TestGene = 1212)"
        elif category == "3SS":
            return "Mouse 3SS (#TestGene = 1212)"
        else:
            return "Mouse 5SS+3SS (#TestGene = 1212)"
    elif dataset == "h":
        if category == "5SS":
            return "Human 5SS (#TestGene = 1629)"
        elif category == "3SS":
            return "Human 3SS (#TestGene = 1629)"
        else:
            return "Human 5SS+3SS (#TestGene = 1629)"
    else:
        return f"{category}: Fraction Correct vs. FB Probability ({dataset.upper()})"

###############################################################################
# 将统计结果记录到文本文件中的函数
###############################################################################
def record_results_to_txt(results_dict, output_filename):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        for key, df in results_dict.items():
            f.write(f"{key} 的结果:\n")
            f.write(df.to_string())
            f.write("\n\n")
    print(f"摘要结果已保存到 {output_filename}")

###############################################################################
# 1) 解析剪接位点文件并提取 Viterbi 信息的函数
###############################################################################
def parse_splice_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    # 注释剪接位点的正则表达式模式
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    # Viterbi（SMsplice）剪接位点的正则表达式模式
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

    # 用于计算正确率的排序预测块正则表达式模式
    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)",
        re.DOTALL
    )
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
    for (pos, prob) in fiveprime_preds:
        is_correct = (pos in annotated_5prime)
        in_viterbi = (pos in viterbi_5prime)
        rows.append((pos, prob, "5prime", is_correct, in_viterbi))
    for (pos, prob) in threeprime_preds:
        is_correct = (pos in annotated_3prime)
        in_viterbi = (pos in viterbi_3prime)
        rows.append((pos, prob, "3prime", is_correct, in_viterbi))
    
    df = pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])
    return df

###############################################################################
# 2) 计算每个概率区间内的统计数据
#    区间：[0.1,0.2), [0.2,0.3), …, [0.8,0.9), [0.9, ∞)
###############################################################################
def compute_stats_per_bin(df, bin_edges):
    df = df.copy()
    df["prob_bin"] = pd.cut(df["prob"], bins=bin_edges, right=False)
    grouped = df.dropna(subset=["prob_bin"]).groupby("prob_bin")
    stats = grouped["is_correct"].agg(["count", "sum", "mean"])
    stats.rename(columns={"count": "N_in_bin", "sum": "N_correct", "mean": "fraction_correct"}, inplace=True)
    bin_intervals = pd.cut(df["prob"], bins=bin_edges, right=False).dtype.categories
    stats = stats.reindex(bin_intervals, fill_value=0)
    return stats

###############################################################################
# 3) 辅助函数：绘制 Viterbi 与非 Viterbi 对比的柱状图
#    x 轴反转，使得最高 FB 概率区间出现在左侧
###############################################################################
def plot_viterbi_comparison(vit_stats, nonvit_stats, bin_edges, output_filename, title):
    bin_intervals = pd.cut(pd.Series([0.0]), bins=bin_edges, right=False).dtype.categories
    bin_labels = [f"{interval.left:.1f}-{interval.right:.1f}" if interval.right != np.inf 
                  else "0.9-1.0" for interval in bin_intervals]
    x = np.arange(len(bin_labels))
    
    vit_frac = vit_stats["fraction_correct"].tolist()
    nonvit_frac = nonvit_stats["fraction_correct"].tolist()
    
    width = 0.35
    plt.figure(figsize=(24,6))
    plt.bar(x - width/2, vit_frac, width=width, color="blue", alpha=0.7, label="Viterbi")
    plt.bar(x + width/2, nonvit_frac, width=width, color="red", alpha=0.7, label="Non-Viterbi")
    
    # 为每个柱子添加标签
    for i in range(len(bin_labels)):
        # Viterbi 的柱子标签
        frac_val_vit = vit_frac[i]
        n_corr_vit = vit_stats["N_correct"].iloc[i]
        n_tot_vit = vit_stats["N_in_bin"].iloc[i]
        plt.text(x[i] - width/2, frac_val_vit + 0.06, f"{frac_val_vit:.2f}", ha="center", va="bottom", fontsize=12)
        plt.text(x[i] - width/2, frac_val_vit + 0.02, f"{int(n_corr_vit)}/{int(n_tot_vit)}", ha="center", va="bottom", fontsize=12)
        
        # Non-Viterbi 的柱子标签
        frac_val_nonvit = nonvit_frac[i]
        n_corr_nonvit = nonvit_stats["N_correct"].iloc[i]
        n_tot_nonvit = nonvit_stats["N_in_bin"].iloc[i]
        plt.text(x[i] + width/2, frac_val_nonvit + 0.06, f"{frac_val_nonvit:.2f}", ha="center", va="bottom", fontsize=12)
        plt.text(x[i] + width/2, frac_val_nonvit + 0.02, f"{int(n_corr_nonvit)}/{int(n_tot_nonvit)}", ha="center", va="bottom", fontsize=12)
    
    plt.xlabel("FB 概率区间", fontsize=16)
    plt.ylabel("正确率", fontsize=16)
    plt.xticks(x, bin_labels, fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=14)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    # 反转 x 轴，使区间降序排列
    plt.gca().invert_xaxis()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"对比图已保存到 {output_filename}")
    plt.close()

###############################################################################
# 4) 新的辅助函数：绘制降序 ECDF（累积正确率）对比图
#    对于阈值（0.9, 0.8, …, 0.1），计算 FB 概率大于等于阈值的预测中的累积正确率
#    在每个阈值处标注比例，下面标注正确预测数和总预测数
###############################################################################
def plot_descending_ecdf_correct_fraction_comparison(vit_df, nonvit_df, output_filename, title):
    # 定义对应区间的阈值：[0.9, 0.8, ..., 0.1]
    thresholds = np.arange(0.9, 0.0, -0.1)
    
    def cumulative_stats(df, threshold):
        subset = df[df["prob"] >= threshold]
        total = len(subset)
        if total == 0:
            return (np.nan, 0, 0)
        correct = subset["is_correct"].sum()
        ratio = correct / total
        return (ratio, correct, total)
    
    vit_results = [cumulative_stats(vit_df, t) for t in thresholds]
    nonvit_results = [cumulative_stats(nonvit_df, t) for t in thresholds]
    
    vit_fractions = [res[0] for res in vit_results]
    nonvit_fractions = [res[0] for res in nonvit_results]
    
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, vit_fractions, marker='o', linestyle='-', label="Viterbi", color="blue")
    plt.plot(thresholds, nonvit_fractions, marker='o', linestyle='-', label="Non-Viterbi", color="red")
    plt.xlabel("≥ FBP", fontsize=16)
    plt.ylabel("Cumulative Correct Fraction", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(thresholds, [f"{t:.1f}-{t+0.1:.1f}" for t in thresholds], fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    # 为每个阈值点标注比例以及正确/总数
    for i, t in enumerate(thresholds):
        # 对于 Viterbi:
        vit_ratio, vit_correct, vit_total = vit_results[i]
        if not np.isnan(vit_ratio):
            plt.text(t, vit_fractions[i] + 0.03, f"{vit_ratio:.2f}", ha="center", va="bottom", color="blue", fontsize=10)
            plt.text(t, vit_fractions[i] - 0.03, f"{int(vit_correct)}/{vit_total}", ha="center", va="top", color="blue", fontsize=10)
        # 对于 Non-Viterbi:
        nonvit_ratio, nonvit_correct, nonvit_total = nonvit_results[i]
        if not np.isnan(nonvit_ratio):
            plt.text(t, nonvit_fractions[i] + 0.03, f"{nonvit_ratio:.2f}", ha="center", va="bottom", color="red", fontsize=10)
            plt.text(t, nonvit_fractions[i] - 0.03, f"{int(nonvit_correct)}/{nonvit_total}", ha="center", va="top", color="red", fontsize=10)
    
    # 反转 x 轴，使较低阈值显示在左侧
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"降序 ECDF（累积正确率）图已保存到 {output_filename}")
    plt.close()

###############################################################################
# 5) 主函数：处理文件并生成 5SS、3SS、合并图以及记录统计结果到文本文件
###############################################################################
def main():
    # 根据数据集选择文件匹配模式
    if dataset == "t":
        pattern = f"./0_t_result/t_result_{top_k}/000_arabidopsis_g_*.txt"
    elif dataset == "z":
        pattern = f"./0_z_result/z_result_{top_k}/000_zebrafish_g_*.txt"
    elif dataset == "m":
        pattern = f"./0_m_result/m_result_{top_k}/000_mouse_g_*.txt"
    elif dataset == "h":
        pattern = f"./0_h_result/h_result_{top_k}/000_human_g_*.txt"
    else:
        print("未知数据集")
        return
    
    files = sorted(glob.glob(pattern))
    num_files = len(files)
    print(f"找到 {num_files} 个与模式 '{pattern}' 匹配的 txt 文件。")
    if num_files == 0:
        print("未找到匹配的文件！")
        return

    all_data = []
    for i, fname in enumerate(files, start=1):
        print(f"\n=== 正在处理文件 {i}/{num_files}: {fname} ===")
        df_splice = parse_splice_file(fname)
        all_data.append(df_splice)
    
    if not all_data:
        print("文件中未找到有效数据。")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    df_5SS = combined_df[combined_df["type"] == "5prime"]
    df_3SS = combined_df[combined_df["type"] == "3prime"]

    # 定义区间边界：[0.1,0.2), [0.2,0.3), …, [0.8,0.9), [0.9, ∞)
    bin_edges = np.array(list(np.arange(0.1, 0.9, 0.1)) + [0.9, np.inf])
    
    # --- Viterbi 与非 Viterbi 的柱状图 ---
    # 处理 5SS 数据（确保 "is_viterbi" 列转换为布尔类型）
    df_5SS_vit = df_5SS[df_5SS["is_viterbi"].astype(bool)]
    df_5SS_nonvit = df_5SS[~df_5SS["is_viterbi"].astype(bool)]
    vit_stats_5 = compute_stats_per_bin(df_5SS_vit, bin_edges)
    nonvit_stats_5 = compute_stats_per_bin(df_5SS_nonvit, bin_edges)
    plot_viterbi_comparison(vit_stats_5, nonvit_stats_5, bin_edges,
                            f"./0_{dataset}_result/v_not_v_5SS_top_{top_k}_{dataset}.png",
                            title=f"{get_title('5SS')} Viterbi vs. Non-Viterbi top_k = {top_k}")
    
    # 处理 3SS 数据
    df_3SS_vit = df_3SS[df_3SS["is_viterbi"].astype(bool)]
    df_3SS_nonvit = df_3SS[~df_3SS["is_viterbi"].astype(bool)]
    vit_stats_3 = compute_stats_per_bin(df_3SS_vit, bin_edges)
    nonvit_stats_3 = compute_stats_per_bin(df_3SS_nonvit, bin_edges)
    plot_viterbi_comparison(vit_stats_3, nonvit_stats_3, bin_edges,
                            f"./0_{dataset}_result/v_not_v_3SS_top_{top_k}_{dataset}.png",
                            title=f"{get_title('3SS')} Viterbi vs. Non-Viterbi top_k = {top_k}")
    
    # 处理所有数据（5SS 和 3SS 合并）
    df_all_vit = combined_df[combined_df["is_viterbi"].astype(bool)]
    df_all_nonvit = combined_df[~combined_df["is_viterbi"].astype(bool)]
    vit_stats_all = compute_stats_per_bin(df_all_vit, bin_edges)
    nonvit_stats_all = compute_stats_per_bin(df_all_nonvit, bin_edges)
    plot_viterbi_comparison(vit_stats_all, nonvit_stats_all, bin_edges,
                            f"./0_{dataset}_result/v_not_v_ALL_top_{top_k}_{dataset}.png",
                            title=f"{get_title('ALL')} Viterbi vs. Non-Viterbi top_k = {top_k}")

    # --- 降序 ECDF（累积正确率）对比图 ---
    plot_descending_ecdf_correct_fraction_comparison(df_5SS_vit, df_5SS_nonvit,
                         f"./0_{dataset}_result/v_not_v_ecdf_5SS_top_{top_k}_{dataset}.png",
                         title=f"{get_title('5SS')}")
    plot_descending_ecdf_correct_fraction_comparison(df_3SS_vit, df_3SS_nonvit,
                         f"./0_{dataset}_result/v_not_v_ecdf_3SS_top_{top_k}_{dataset}.png",
                         title=f"{get_title('3SS')}")
    plot_descending_ecdf_correct_fraction_comparison(df_all_vit, df_all_nonvit,
                         f"./0_{dataset}_result/v_not_v_ecdf_ALL_top_{top_k}_{dataset}.png",
                         title=f"{get_title('ALL')}")
    
    # --- 将统计结果记录到文本文件 ---
    summary_results = {
        "5SS Viterbi 统计": vit_stats_5,
        "5SS Non-Viterbi 统计": nonvit_stats_5,
        "3SS Viterbi 统计": vit_stats_3,
        "3SS Non-Viterbi 统计": nonvit_stats_3,
        "ALL Viterbi 统计": vit_stats_all,
        "ALL Non-Viterbi 统计": nonvit_stats_all,
    }
    output_txt = f"./0_{dataset}_result/v_not_v_ecdf_top_{top_k}_{dataset}.txt"
    record_results_to_txt(summary_results, output_txt)
    
if __name__ == "__main__":
    main()
