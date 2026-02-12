"""
模块4：高/低压分类 (SVM增强版)
功能：特征提取 + PCA降维 + SVM分类 + 中文混淆矩阵可视化
说明：
- 统一特征：对每个保留通道提取 6 个统计特征（均值/标准差/RMS/峰峰值/偏度/峭度）
- 通道处理：剔除 0,4,8(高���电压通道) 和 12-17(绕组通道)，剩余 18 个通道
- 与模块3保持同一套绘图字体、字号、标注格式（数量 + 占总样本百分比）
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 设置中文字体 (解决Matplotlib中文乱码)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
warnings.filterwarnings("ignore")

# ==================== 配置参数（与模块3一致风格） ====================
OUTPUT_PATH = "output"
MODEL_DIR = "models"
FIGURE_DIR = "figures"

# 剔除通道：0/4/8（高压电压通道），12-17（绕组通道）
DROP_CHANNELS = [0, 4, 8, 12, 13, 14, 15, 16, 17]

# [可调参数]（给出参考值）
PCA_VARIANCE = 0.99  # PCA保留方差比例（建议：0.95 ~ 0.99）
SVM_C = 10  # SVM惩罚系数（建议：0.1, 1, 10, 100）
SVM_GAMMA = "scale"  # 核函数系数（建议：'scale' 或 'auto'；也可手动如 0.01）
SVM_CLASS_WEIGHT = None  # 类别不均衡时可设为 'balanced'；均衡数据可用 None
SVM_KERNEL = "rbf"  # 建议 rbf；若想对比可试 'linear'

# [可调参数]：混淆矩阵配色（参考你图1：绿色系）
USE_GREEN_CMAP = True  # True：绿；False：蓝（模块3默认Blues）
GREEN_CMAP_NAME = "Greens"  # 你也可以改成 'YlGn' / 'GnBu' 等

RANDOM_STATE = 42


def create_directories():
    for d in [MODEL_DIR, FIGURE_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)


def extract_features(data: np.ndarray) -> np.ndarray:
    """
    输入 data: shape = (T, C)  (模块1保存的npy)
    输出 feature_vector: shape = (C_kept * 6, )
    """
    all_indices = np.arange(data.shape[1])
    keep_indices = np.setdiff1d(all_indices, DROP_CHANNELS)
    filtered_data = data[:, keep_indices]

    feature_vector = []
    for col in range(filtered_data.shape[1]):
        signal = filtered_data[:, col]

        f_std = np.std(signal)
        if f_std > 1e-6:
            f_skew = stats.skew(signal)
            f_kurt = stats.kurtosis(signal)
        else:
            # 避免全常数信号导致数值问题
            f_skew = 0
            f_kurt = 0

        feature_vector.extend(
            [
                np.mean(signal),
                f_std,
                np.sqrt(np.mean(signal**2)),  # RMS
                np.ptp(signal),  # peak-to-peak
                f_skew,
                f_kurt,
            ]
        )
    return np.array(feature_vector, dtype=float)


def load_data(csv_path: str):
    """
    读取模块2输出的 train_dataset.csv / test_dataset.csv
    其中必须包含：
    - file_path：npy路径
    - voltage_binary：高低压二分类标签（约定：0高压，1低压；与模块1保持一致）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    X_list, y_list = [], []

    for _, row in df.iterrows():
        fp = row["file_path"]
        if os.path.exists(fp):
            try:
                data = np.load(fp)
                X_list.append(extract_features(data))
                y_list.append(int(row["voltage_binary"]))
            except Exception:
                # 与模块3一致：坏样本静默跳过
                pass

    return np.array(X_list), np.array(y_list), df


def plot_confusion_matrix_custom(y_true, y_pred, title, filename, labels):
    """绘制美观的混淆矩阵 (百分比基准：总样本数) + 模块3同款字体/字号/标注格式"""
    cm = confusion_matrix(y_true, y_pred)

    # 百分比按“总样本数”计算（与你要求一致：当前数值/总样本数）
    total_samples = cm.sum()
    cm_percent = cm.astype("float") / total_samples * 100

    plt.figure(figsize=(8, 6))

    # 自定义注释：数量 + (占总数百分比)
    annot_labels = [
        f"{val}\n({pct:.1f}%)" for val, pct in zip(cm.flatten(), cm_percent.flatten())
    ]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)

    cmap = GREEN_CMAP_NAME if USE_GREEN_CMAP else "Blues"

    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14},  # 模块3同款：数字字号
        cbar=True,
    )

    plt.xlabel("预测结果", fontsize=12)
    plt.ylabel("真实标签", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"可视化图表已保存: {save_path}")
    plt.close()


def main():
    create_directories()

    # 1. 加载数据
    train_csv = os.path.join(OUTPUT_PATH, "train_dataset.csv")
    test_csv = os.path.join(OUTPUT_PATH, "test_dataset.csv")

    X_train, y_train, _ = load_data(train_csv)
    X_test, y_test, _ = load_data(test_csv)

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("训练集或测试集为空，请检查CSV路径、npy文件路径与数据有效性。")

    # 2. 模型流水线（与模块3一致：Scaler + PCA + SVM）
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)),
            (
                "svm",
                SVC(
                    kernel=SVM_KERNEL,
                    C=SVM_C,
                    gamma=SVM_GAMMA,
                    probability=True,
                    class_weight=SVM_CLASS_WEIGHT,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 3. 训练
    print(
        f"\n开始训练高/低压分类 (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA}, "
        f"class_weight={SVM_CLASS_WEIGHT}, PCA={PCA_VARIANCE})..."
    )
    pipeline.fit(X_train, y_train)
    print(f"PCA保留特征数: {pipeline.named_steps['pca'].n_components_}")

    # 4. 预测
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # 预测为“低压(1)”的概率

    # 5. 评估与可视化
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=["高压", "低压"]))

    plot_confusion_matrix_custom(
        y_test,
        y_pred,
        title=f"高、压分类结果混淆矩阵\n(准确率={acc:.1%})",
        filename="HV_LV_confusion_matrix.png",
        labels=["高压", "低压"],
    )

    # 6. 保存模型与结果
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "svm_HVLV_voltage.pkl"))

    results_df = pd.DataFrame(
        {"true_label": y_test, "pred_label": y_pred, "pred_prob_LV": y_prob}
    )
    results_df.to_csv(
        os.path.join(OUTPUT_PATH, "HVLV_classification_results.csv"), index=False
    )

    print("\n模型已保存至:", os.path.join(MODEL_DIR, "svm_HVLV_voltage.pkl"))
    print("结果已保存至:", os.path.join(OUTPUT_PATH, "HVLV_classification_results.csv"))


if __name__ == "__main__":
    main()