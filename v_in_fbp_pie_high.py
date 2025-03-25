#!/usr/bin/env python3
import sys
import re
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 物種代碼對應全名
species_names = {
    "t": "Arabidopsis",
    "z": "Zebrafish",
    "m": "Mouse",
    "h": "Human",
    "f": "Fly",
    "o": "Moth"
}

# 檢查命令列參數：dataset 與 top_k 與 seed
if len(sys.argv) < 4:
    print("Usage: script.py <dataset> <top_k> <seed>")
    sys.exit(1)
dataset_arg = sys.argv[1]   # 若指定 "all" 則處理所有物種
top_k = int(sys.argv[2])
my_seed = int(sys.argv[3])
print(my_seed)


def parse_splice_file(filename):
    """
    解析一個文本文件，返回：
      - df: 包含排序預測的 DataFrame（每行記錄位置、概率、類型、是否正確、是否屬於 SMsplice/Viterbi）
      - annotated_5: Annotated 5' 剪接位點（集合，GT）
      - annotated_3: Annotated 3' 剪接位點（集合，GT）
      - viterbi_5: SMsplice 5' 预测（集合）
      - viterbi_3: SMsplice 3' 预测（集合）
    """
    with open(filename, "r") as f:
        text = f.read()

    # 解析 Annotated 位點（真實剪接位點 GT）
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    # 解析 SMsplice（Viterbi預測）位點
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

    # 解析排序預測部分（5' 與 3' 分開）
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
    根據 dataset 和 top_k 值，返回匹配的文件列表
    """
    if dataset == "t":
        pattern = f"./{my_seed}_t_result/t_result_{top_k_val}/000_arabidopsis_g_*.txt"
    elif dataset == "z":
        pattern = f"./{my_seed}_z_result/z_result_{top_k_val}/000_zebrafish_g_*.txt"
    elif dataset == "m":
        pattern = f"./{my_seed}_m_result/m_result_{top_k_val}/000_mouse_g_*.txt"
    elif dataset == "h":
        pattern = f"./{my_seed}_h_result/h_result_{top_k_val}/000_human_g_*.txt"
    elif dataset == "f":
        pattern = f"./{my_seed}_f_result/f_result_{top_k_val}/000_fly_g_*.txt"
    elif dataset == "o":
        pattern = f"./{my_seed}_o_result/o_result_{top_k_val}/000_moth_g_*.txt"
    else:
        print("未知 dataset")
        return []
    return sorted(glob.glob(pattern))


def calc_metrics_from_counts(tp, total_pred, total_gt):
    """
    根據累計的真陽性、預測總數和真實總數計算 Recall, Precision, F1
    """
    recall = tp / total_gt if total_gt > 0 else 0
    precision = tp / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1


def get_bin_info(files, threshold):
    """
    將所有文件中 Viterbi 預測（過濾條件： prob >= threshold）彙總，
    並計算每個概率區間的統計資訊。
    
    返回一個 DataFrame，包含：
      - N_in_bin: 每個區間的預測數量
      - N_correct: 每個區間中正確預測的數量
      - Precision: 正確率（百分比）
      - percentage: 區間預測數量占總數的百分比
    並在末尾附加 "Total" 行。
    """
    all_preds = []  # (prob, is_correct)
    for fname in files:
        df, annotated_5, annotated_3, viterbi_5, viterbi_3 = parse_splice_file(fname)
        gt = set(list(annotated_5) + list(annotated_3))
        # 處理 5' 預測
        for pos in viterbi_5:
            rows = df[(df["position"] == pos) & (df["type"] == "5prime")]
            if not rows.empty:
                prob = rows["prob"].iloc[0]
                if prob >= threshold:
                    is_correct = pos in gt
                    all_preds.append((prob, is_correct))
            else:
                prob = -0.5  # 未找到對應位置，指定落在 [-1,0) 區間
                is_correct = pos in gt
                all_preds.append((prob, is_correct))
        # 處理 3' 預測
        for pos in viterbi_3:
            rows = df[(df["position"] == pos) & (df["type"] == "3prime")]
            if not rows.empty:
                prob = rows["prob"].iloc[0]
                if prob >= threshold:
                    is_correct = pos in gt
                    all_preds.append((prob, is_correct))
            else:
                prob = -0.5
                is_correct = pos in gt
                all_preds.append((prob, is_correct))
    preds_df = pd.DataFrame(all_preds, columns=["prob", "is_correct"])
    # 將預測值超過 1.0 的 clip 至 1.0，確保頂區間計算正確
    preds_df["prob"] = preds_df["prob"].clip(upper=0.99)
    
    # 定義區間，包含 [-1,0) 區間，並將頂區間分為兩個區間: [0.9,0.95) 和 [0.95, 1.0]
    bin_edges = np.array(
        [-1.0, 0.0] +
        list(np.arange(0.1, 0.9, 0.1)) +
        [0.9, 0.95, 1.0]
    )
    bin_labels = [
        "[-1,0)", "(0.0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)",
        "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)",
        "[0.9,0.95)", "[0.95, 1.0]"
    ]
    preds_df["bin"] = pd.cut(preds_df["prob"], bins=bin_edges, labels=bin_labels, right=False)
    bin_info = preds_df.groupby("bin").agg(
        N_in_bin=("prob", "count"),
        N_correct=("is_correct", "sum")
    )
    # 補全所有區間
    bin_info = bin_info.reindex(bin_labels, fill_value=0)
    bin_info["Precision"] = (bin_info["N_correct"] / bin_info["N_in_bin"]).fillna(0) * 100

    # 計算總數，並新增百分比欄位：每個區間的數量占總數的百分比
    total_N_in_bin = bin_info["N_in_bin"].sum()
    bin_info["percentage"] = bin_info["N_in_bin"] / total_N_in_bin * 100

    # 將 bin 倒序排列（不包括 Total 行）
    bin_info = bin_info.iloc[::-1]
    total_N_correct = bin_info["N_correct"].sum()
    total_fraction = 100 * (total_N_correct / total_N_in_bin) if total_N_in_bin > 0 else 0
    total_row = pd.DataFrame({
        "N_in_bin": [total_N_in_bin],
        "N_correct": [total_N_correct],
        "Precision": [total_fraction],
        "percentage": [100]
    }, index=["Total"])
    bin_info = pd.concat([bin_info, total_row])
    return bin_info


def save_bin_info_to_csv(dataset, top_k_val, threshold):
    """
    將指定 threshold 下的 bin 統計資訊存入 CSV 檔
    """
    files = get_files_for_topk(dataset, top_k_val)
    bin_info_df = get_bin_info(files, threshold)
    outcsv = f"../{my_seed}_{dataset}_top_{top_k_val}.csv"
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    bin_info_df.to_csv(outcsv)
    print(f"Bin info saved to {outcsv}")


# ----------------- 繪製圖形 -----------------
# 定義 bin 與顏色的映射（排除 "[-1,0)" 的 bin）
bin_color_map = {
    "[0.95, 1.0]": "#87CEFA",   # 水藍
    "[0.9,0.95)": "#90EE90",     # 淺綠
    "[0.8,0.9)": "#FFA07A",      # 淺鮭魚橘
    "[0.7,0.8)": "#FFD700",      # 金色
    "[0.6,0.7)": "#DDA0DD",      # 梅紫
    "[0.5,0.6)": "#20B2AA",      # 淺海綠
    "[0.4,0.5)": "#FF69B4",      # 熱粉
    "[0.3,0.4)": "#A9A9A9",      # 灰色
    "[0.2,0.3)": "#FFB6C1",      # 粉紅
    "(0.0,0.1)": "#D3D3D3"       # 淺灰（用於 0.0-0.1 區間）
}


def plot_species_pie(ax, species_name, overall_precision, bins_labels, precision_vals, counts):
    # 排除僅「[-1,0)」的區間，保留其他區間（包括 "(0.0,0.1)"）
    valid_data = [(lbl, pv, ct) for lbl, pv, ct in zip(bins_labels, precision_vals, counts)
                  if ct > 0 and lbl != "[-1,0)"]
    if not valid_data:
        ax.set_title(f"{species_name}\n(No data)")
        return
    wedge_counts = []
    wedge_colors = []
    wedge_labels = []
    wedge_precisions = []
    for lbl, pv, ct in valid_data:
        color = bin_color_map.get(lbl, "#D3D3D3")
        wedge_counts.append(ct)
        wedge_colors.append(color)
        wedge_labels.append(lbl)
        wedge_precisions.append(pv)
    # 繪製圓餅圖，不直接在 slice 上顯示百分比
    wedges, _ = ax.pie(
        wedge_counts, labels=None, startangle=140, colors=wedge_colors
    )
    total_wedge = sum(wedge_counts)
    # legend 中加入區間資訊：標籤、Precision 與該區間數量占比
    legend_labels = []
    for lbl, pv, ct in zip(wedge_labels, wedge_precisions, wedge_counts):
        pct = ct / total_wedge * 100
        legend_labels.append(f"{lbl}: {pv:.2f} ( {pct:.1f}% )")
    ax.legend(wedges, legend_labels, title="Bins and Precision", loc="center left", bbox_to_anchor=(1, 0.5))
    # 在圓餅圖中央顯示總體 Precision
    ax.text(0, 0, f"Overall\n{overall_precision:.2f}%", fontsize=14, fontweight="bold", ha="center")
    ax.set_title(species_name)


def plot_species_bar(ax, bins_labels, precision_vals):
    """
    繪製直方圖，x 軸為 bin 標籤，y 軸為 precision 值（百分比）
    """
    ax.bar(bins_labels, precision_vals, color=[bin_color_map.get(lbl, "#D3D3D3") for lbl in bins_labels])
    ax.set_xlabel("Bins")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Bin Precision")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_all_species_pies(top_k_val, threshold=0.00):
    # 構造各 species 資料
    species_data = {}
    for code, full_name in species_names.items():
        files = get_files_for_topk(code, top_k_val)
        bin_df = get_bin_info(files, threshold)
        overall_precision = bin_df.loc["Total", "Precision"]
        # 移除 Total 行
        bin_df = bin_df.drop("Total", errors="ignore")
        bins_labels = list(bin_df.index)
        precision_vals = list(bin_df["Precision"])
        counts = list(bin_df["N_in_bin"])
        species_data[full_name] = {
            "overall_precision": overall_precision,
            "bins_labels": bins_labels,
            "precision_values": precision_vals,
            "counts": counts
        }
    # 使用 2x4 的網格，每個 species 佔用一行兩個子圖（左側圓餅圖、右側直方圖），共 4 個 species
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    axs = axs.flatten()
    species_list = list(species_data.items())
    for i, (species, data) in enumerate(species_list):
        pie_ax = axs[i*2]
        bar_ax = axs[i*2 + 1]
        plot_species_pie(pie_ax, species, data["overall_precision"],
                         data["bins_labels"], data["precision_values"], data["counts"])
        # 直方圖：僅排除 "[-1,0)" 區間
        valid_bins = [lbl for lbl in data["bins_labels"] if lbl != "[-1,0)"]
        valid_precisions = [pv for lbl, pv in zip(data["bins_labels"], data["precision_values"])
                            if lbl != "[-1,0)"]
        plot_species_bar(bar_ax, valid_bins, valid_precisions)
    fig.suptitle("Distribution of Precision by Bins for Different Species", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    plt.savefig(f"v_in_fbp_pie_top_{top_k}_seed_{my_seed}.png")
    print(f"Pie and bar charts saved to precision_distribution_by_bins_{top_k}.png")


def main():
    # 若 dataset 參數不是 "all"，則只處理該單一物種並存出 CSV
    if dataset_arg != "all":
        files = get_files_for_topk(dataset_arg, top_k)
        bin_info_df = get_bin_info(files, 0.00)
        print("Bin information for predictions with prob >= 0.00:")
        print(bin_info_df)
        save_bin_info_to_csv(dataset_arg, top_k, 0.00)
    else:
        # 若指定 "all"，則針對所有物種處理，並各自存出 CSV
        for code in species_names.keys():
            bin_info_df = get_bin_info(get_files_for_topk(code, top_k), 0.00)
            save_bin_info_to_csv(code, top_k, 0.00)
    # 繪製所有物種圖形：每個 species 左側圓餅圖、右側直方圖
    # plot_all_species_pies(top_k, 0.00)


if __name__ == "__main__":
    main()
