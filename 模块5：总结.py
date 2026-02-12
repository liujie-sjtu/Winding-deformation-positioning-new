"""
模块5：简单组合融合（适配new仓库）
功能：
1) 读取模块3(AB相二分类)与模块4(高低压二分类)的测试集预测结果CSV
2) 校验样本一致性
3) 组合得到4类分类结果：A高/A低/B高/B低
4) 输出分类报告 + 绘制紫色(Purples)四分类混淆矩阵（数量 + 占总样本百分比，黑白自适应）

依赖输入（默认路径）：
- output/AB_classification_results.csv           （模块3输出）
- output/HVLV_classification_results.csv         （模块4输出）
可在参数区修改文件名以适配你的实际输出。

注意：
- 约定：phase_binary: 0=A相, 1=B相
- 约定：voltage_binary: 0=高压, 1=低压
- 4类编码：phase*2 + voltage
"""

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# =============== Matplotlib中文 ===============
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置参数 ====================
OUTPUT_PATH = "output"
FIGURE_DIR = "figures"

# 模块3/模块4的结果文件（如你实际文件名不同，在这里改）
AB_RESULTS_FILE = os.path.join(OUTPUT_PATH, "AB_classification_results.csv")
VOLTAGE_RESULTS_FILE = os.path.join(OUTPUT_PATH, "HVLV_classification_results.csv")

# 输出文件
FUSION_RESULTS_CSV = os.path.join(OUTPUT_PATH, "final_4class_fusion_results.csv")
FUSION_CONFUSION_FIG = os.path.join(FIGURE_DIR, "fusion_4class_confusion_matrix.png")

# 4类标签映射
LABEL_MAPPING = {
    0: "A相高压",
    1: "A相低压",
    2: "B相高压",
    3: "B相低压",
}


def create_directories():
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)


def load_results():
    if not os.path.exists(AB_RESULTS_FILE):
        raise FileNotFoundError(f"未找到模块3结果文件: {AB_RESULTS_FILE}")
    if not os.path.exists(VOLTAGE_RESULTS_FILE):
        raise FileNotFoundError(f"未找到模块4结果文件: {VOLTAGE_RESULTS_FILE}")

    ab = pd.read_csv(AB_RESULTS_FILE)
    vol = pd.read_csv(VOLTAGE_RESULTS_FILE)

    # 兼容不同列名：尽量自动识别“真值列/预测列”
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    ab_true_col = find_col(ab, ["true_label", "true", "y_true"])
    ab_pred_col = find_col(ab, ["pred_label", "predicted_label", "y_pred"])
    vol_true_col = find_col(vol, ["true_label", "true", "y_true"])
    vol_pred_col = find_col(vol, ["pred_label", "predicted_label", "y_pred"])

    if ab_true_col is None or ab_pred_col is None:
        raise ValueError(f"模块3结果列名不符合预期，当前列: {list(ab.columns)}")
    if vol_true_col is None or vol_pred_col is None:
        raise ValueError(f"模块4结果列名不符合预期，当前列: {list(vol.columns)}")

    # 统一列名，便于后续处理
    ab = ab.rename(columns={ab_true_col: "ab_true", ab_pred_col: "ab_pred"})
    vol = vol.rename(columns={vol_true_col: "vol_true", vol_pred_col: "vol_pred"})

    return ab, vol


def validate_consistency(ab: pd.DataFrame, vol: pd.DataFrame):
    """
    尽可能保证两个结果文件行顺序一致。
    - 若两个文件都含 file_path，则按 file_path 对齐
    - 否则按行号强制对齐（要求两者长度一致）
    """
    ab_has_fp = "file_path" in ab.columns
    vol_has_fp = "file_path" in vol.columns

    if ab_has_fp and vol_has_fp:
        merged = pd.merge(
            ab[["file_path", "ab_true", "ab_pred"]],
            vol[["file_path", "vol_true", "vol_pred"]],
            on="file_path",
            how="inner",
        )
        if len(merged) == 0:
            raise ValueError("按 file_path 对齐失败：交集为空，请检查两个结果文件是否对应同一批测试样本。")
        return merged

    # 无 file_path：要求长度一致
    if len(ab) != len(vol):
        raise ValueError(f"结果文件长度不一致：模块3={len(ab)}，模块4={len(vol)}，且缺少 file_path 无法对齐。")

    merged = pd.DataFrame(
        {
            "ab_true": ab["ab_true"].values,
            "ab_pred": ab["ab_pred"].values,
            "vol_true": vol["vol_true"].values,
            "vol_pred": vol["vol_pred"].values,
        }
    )
    return merged


def build_4class_labels(df_merged: pd.DataFrame):
    """
    4类编码：phase*2 + voltage
    phase: 0(A),1(B)
    voltage: 0(高),1(低)
    """
    y_true_4 = df_merged["ab_true"].astype(int) * 2 + df_merged["vol_true"].astype(int)
    y_pred_4 = df_merged["ab_pred"].astype(int) * 2 + df_merged["vol_pred"].astype(int)
    return y_true_4.values, y_pred_4.values


def plot_confusion_matrix_4class(y_true, y_pred, title, filename):
    """
    紫色四分类混淆矩阵：
    - 每格显示：数量 + (占总样本百分比)
    - 百分比文字黑白自适应：深色底白字，浅色底黑字
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    total = cm.sum()
    pct = (cm.astype(float) / total * 100) if total > 0 else np.zeros_like(cm, dtype=float)

    labels = [LABEL_MAPPING[i] for i in [0, 1, 2, 3]]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 先画热图，但不让 seaborn 写注释，我们自己写（为了控制黑白自适应）
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("预测结果", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)

    # 计算“深色/浅色”阈值：用当前图的颜色范围中值来判断
    vmax = cm.max() if cm.size else 0
    threshold = vmax * 0.5

    # 模块3同款：注释��号 14；文本格式：数量 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            p = pct[i, j]
            text = f"{val}\n({p:.1f}%)"

            # 深色底用白字，浅色底用深色字
            color = "white" if val > threshold else "#111111"

            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=14,
                color=color,
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"四分类混淆矩阵已保存: {filename}")
    plt.close(fig)


def main():
    print("==================== 模块5：简单组合融合（四分类） ====================")
    create_directories()

    ab, vol = load_results()
    merged = validate_consistency(ab, vol)

    y_true_4, y_pred_4 = build_4class_labels(merged)

    print("\n4类真实标签分布:")
    dist = Counter(y_true_4)
    for k in [0, 1, 2, 3]:
        print(f"  {LABEL_MAPPING[k]}: {dist.get(k, 0)} 个样本")

    acc = accuracy_score(y_true_4, y_pred_4)
    print(f"\n四分类测试集准确率: {acc:.2%}\n")
    print(classification_report(y_true_4, y_pred_4, target_names=[LABEL_MAPPING[i] for i in [0, 1, 2, 3]]))

    # 保存融合结果
    out_df = merged.copy()
    out_df["true_4class"] = y_true_4
    out_df["pred_4class"] = y_pred_4
    out_df["true_4class_text"] = [LABEL_MAPPING[i] for i in y_true_4]
    out_df["pred_4class_text"] = [LABEL_MAPPING[i] for i in y_pred_4]
    out_df["correct"] = (y_true_4 == y_pred_4)
    out_df.to_csv(FUSION_RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n融合结果已保存: {FUSION_RESULTS_CSV}")

    # 绘图（紫色、无红色百分比、黑白自适应）
    plot_confusion_matrix_4class(
        y_true_4,
        y_pred_4,
        title=f"四分类故障识别结果混淆矩阵\n(准确率={acc:.1%})",
        filename=FUSION_CONFUSION_FIG,
    )

    print("==================== 模块5 执行完毕 ====================")


if __name__ == "__main__":
    main()