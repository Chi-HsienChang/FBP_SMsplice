import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ======================
# 1. 讀入 CSV 並處理表頭
# ======================
file_path = "Z.csv"  # <-- 修改為你的 CSV 路徑/檔名
raw_df = pd.read_csv(file_path)

# 取出第一列（通常是 "Precision" / "percentage"）資訊
header_row = raw_df.iloc[0]

# 找出「物種名稱」的位置：'Precision' 出現的欄位
species_positions = [i for i, val in enumerate(header_row) if val == 'Precision']
# 物種名稱對應的欄位在 'Precision' 欄位的前一個欄位
species_labels = [raw_df.columns[i - 1] for i in species_positions]

# 從第二列開始才是實際資料
clean_df = raw_df.iloc[1:].copy()

# 將前兩列合併成單一欄位名稱：例如 "Precision_k = 100"、"percentage_k = 100.1"
clean_df.columns = [f"{a}_{b}" for a, b in zip(raw_df.iloc[0], raw_df.columns)]
clean_df = clean_df.reset_index(drop=True)

# 自動找出 bin label 欄位（包含 "Unnamed: 1"），並將其重新命名為 "Zebrafish"
bin_label_col = [col for col in clean_df.columns if "Zebrafish" in col][0]
clean_df.rename(columns={bin_label_col: "Zebrafish"}, inplace=True)
bin_labels = clean_df["Zebrafish"]

# ======================
# 2. 顏色對應表（用於 Pie Chart）
# ======================
bin_color_map = {
    "[0.9, 1.0]": "#2894FF",   # 水藍
    "[0.8,0.9)": "#87CEFA",    # 亮綠

    "[0.7,0.8)": "#FFA07A",    # 淺鮭魚橘
    "[0.6,0.7)": "#FFC78E",    # 金色

    "[0.5,0.6)": "#20B2AA",    # 梅紫
    "[0.4,0.5)": "#90EE90",    # 淺海綠

    "[0.3,0.4)": "#9F35FF",    # 熱粉
    "[0.2,0.3)": "#CA8EFF",    # 灰色

    "[0.1,0.2)": "#FFBFFF"     # 粉紅
}

def get_bin_color(lbl):
    """Return the overridden color for a bin label if applicable,
    otherwise use bin_color_map."""
    lbl_clean = lbl.strip()
    if lbl_clean in ["(0.0,0.1)", "[0.0,0.1)"]:
        return "#BEBEBE"  # black (here represented as gray)
    elif lbl_clean in ["0.0", "0", "[0.0,0.0)", "[0.0,0.0)", "[-1,0)"]:
        return "#FF0000"  # red
    else:
        return bin_color_map.get(lbl_clean, "#FF0000")

# ======================
# 3. 繪圖函式
# ======================
def plot_species_all_k(species_label):
    """
    針對指定的物種（species_label），繪製：
      1. k=100 的 Pie
      2. k=500 的 Pie
      3. k=1000 的 Pie
      4. Bar Chart（一次比較 k=100, k=500, k=1000 的 Precision）
    """
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    
    # 過濾掉 "[-1,0)" 與 "Total" 的 bin
    valid_mask = (bin_labels != "[-1,0)") & (bin_labels != "Total")
    bins = bin_labels[valid_mask]
    
    # 為 pie chart 使用對應顏色（依照 get_bin_color override）
    pie_colors = [get_bin_color(lbl) for lbl in bins]

    # 畫每個 k 值的 Pie Chart
    for i, k in enumerate(["100", "500", "1000"]):
        precision_col = f"Precision_k = {k}"
        percentage_col = f"percentage_k = {k}.1"
        precision_vals = clean_df[precision_col].astype(float)
        percentage_vals = clean_df[percentage_col].astype(float)
        prec = precision_vals[valid_mask]
        perc = percentage_vals[valid_mask]
        
        wedges, _ = axs[i].pie(perc, colors=pie_colors, startangle=140)
        legend_labels = [
            f"{lbl}: {p:.1f} ({c:.1f}%)"
            for lbl, p, c in zip(bins, prec, perc)
        ]
        # 移動 Pie Chart 軸向上（offset 可根據需要調整）
        pos = axs[i].get_position()
        axs[i].set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])
        
        # Legend 放在圖形正下方
        axs[i].legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.28),
                      fontsize=8, ncol=2)
        axs[i].set_title(f"Zebrafish\n$k={k}$", fontsize=20)
    
    # -------------------------
    # Bar Chart
    # -------------------------
    # 直方圖使用與 Pie Chart 相同的填充色
    bar_fill_colors = [get_bin_color(lbl) for lbl in bins]
    
    # 定義不同 k 值的邊線風格：不同的 hatch pattern與邊線顏色
    edge_styles = {
        "100": {"edgecolor": "black", "linewidth": 1.5, "hatch": ""},
        "500": {"edgecolor": "black", "linewidth": 1.5, "hatch": "-"},
        "1000": {"edgecolor": "black", "linewidth": 1.5, "hatch": "//"}
    }
    
    bar_ax = axs[3]
    bar_width = 0.25
    x = range(len(bins))
    
    for i, k in enumerate(["100", "500", "1000"]):
        col = f"Precision_k = {k}"
        vals = clean_df[col].astype(float)[valid_mask]
        style = edge_styles[k]
        bar_ax.bar([xi + i * bar_width for xi in x], vals,
                   width=bar_width, label=f"$k={k}$",
                   color=bar_fill_colors,
                   edgecolor=style["edgecolor"],
                   linewidth=style["linewidth"],
                   hatch=style["hatch"])
    
    bar_ax.set_xticks([xi + bar_width for xi in x])
    bar_ax.set_xticklabels(bins, rotation=45, ha="right")
    bar_ax.set_ylim(0, 100)
    bar_ax.set_title(f"Precision by Bin", fontsize=20)
    bar_ax.set_ylabel("Precision")
    # 建立 legend，只展示邊框 + hatch，不顯示 facecolor（填色）
    legend_handles = []
    
    for k in ["100", "500", "1000"]:
        style = edge_styles[k]
        patch = Patch(facecolor="none",  # 無填色
                      edgecolor=style["edgecolor"],
                      hatch=style["hatch"],
                      linewidth=style["linewidth"],
                      label=f"$k={k}$")
        legend_handles.append(patch)

    bar_ax.legend(handles=legend_handles, loc="upper right")
    
    # fig.suptitle(f"{species_label} - Score Distribution and Precision", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./FBP_SMsplice/0_{species_label}_plot.png")
    plt.show()

# ======================
# 測試繪圖（單一物種）
# ======================
plot_species_all_k("Zebrafish")
