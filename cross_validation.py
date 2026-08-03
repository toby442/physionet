import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, precision_score,
    confusion_matrix, accuracy_score  # <-- 新增 accuracy_score
)
from imblearn.under_sampling import TomekLinks

# ===== 合并训练集+测试集，构成完整的干净数据集(共8279条)，用于10折交叉验证 =====
X_train = np.load("./processed_data/train_embeddings.npy")
X_test = np.load("./processed_data/test_embeddings.npy")
y_train = pd.read_csv("./processed_data/train_labels.csv")["Predict"]
y_test = pd.read_csv("./processed_data/test_labels.csv")["Predict"]

X_full = np.vstack([X_train, X_test])
y_full = pd.concat([y_train, y_test], ignore_index=True)

print(f"完整数据集: {X_full.shape}, 标签分布: {y_full.value_counts().to_dict()}")

# ===== 10折分层交叉验证设置 =====
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def run_cv(method_name, resample_func=None):
    """
    对指定方法跑10折交叉验证
    resample_func: 如果不是None，就在每一折的训练集内部做重采样（避免数据泄漏）
    """
    fold_recalls, fold_f1s, fold_aucs, fold_precisions, fold_accuracies = [], [], [], [], []  # <-- 新增 fold_accuracies
    fold_confusion_matrices = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        # ===== 新增：验证分层是否正确 —— 打印本折训练/验证集里正负样本各自的数量 =====
        train_pos = (y_tr == "Y").sum()
        train_neg = (y_tr == "N").sum()
        val_pos = (y_val == "Y").sum()
        val_neg = (y_val == "N").sum()
        print(f"  [Fold {fold_idx} 分层检查] Train: Y={train_pos}, N={train_neg} "
              f"(共{len(y_tr)}, Y占比{train_pos/len(y_tr)*100:.2f}%) | "
              f"Val: Y={val_pos}, N={val_neg} (共{len(y_val)}, Y占比{val_pos/len(y_val)*100:.2f}%)")

        # 标准化：只用训练折数据fit，避免验证折信息泄漏
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # 如果指定了重采样方法，只在训练折内部做（验证折永远是原始真实数据）
        if resample_func is not None:
            X_tr_scaled, y_tr = resample_func(X_tr_scaled, y_tr)

        # 训练SVM
        if method_name == "Class Weight Adjustment":
            svm = SVC(kernel="linear", class_weight="balanced", random_state=42)
        else:
            svm = SVC(kernel="linear", random_state=42)  # 已重采样平衡，不需要class_weight

        svm.fit(X_tr_scaled, y_tr)
        y_pred = svm.predict(X_val_scaled)
        y_score = svm.decision_function(X_val_scaled)
        y_val_binary = (y_val == "Y").astype(int)

        recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
        precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
        auc = roc_auc_score(y_val_binary, y_score)
        accuracy = accuracy_score(y_val, y_pred)  # <-- 新增 accuracy 计算

        # 混淆矩阵
        cm = confusion_matrix(y_val, y_pred, labels=["N", "Y"])
        tn, fp, fn, tp = cm.ravel()
        fold_confusion_matrices.append({
            "Fold": fold_idx,
            "TN (Not Gout correct)": tn,
            "FP (Gout misclassified)": fp,
            "FN (Gout missed)": fn,
            "TP (Gout correct)": tp
        })

        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_precisions.append(precision)
        fold_aucs.append(auc)
        fold_accuracies.append(accuracy)  # <-- 新增

        print(f"  Fold {fold_idx}: Accuracy={accuracy:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, "
              f"F1={f1:.3f}, AUC={auc:.3f}, CM(TN={tn}, FP={fp}, FN={fn}, TP={tp})")

    print(f"\n===== {method_name} — 10折平均结果 =====")
    print(f"Accuracy:  {np.mean(fold_accuracies):.3f} (std={np.std(fold_accuracies):.3f})")  # <-- 新增
    print(f"Recall:    {np.mean(fold_recalls):.3f} (std={np.std(fold_recalls):.3f})")
    print(f"Precision: {np.mean(fold_precisions):.3f} (std={np.std(fold_precisions):.3f})")
    print(f"F1:        {np.mean(fold_f1s):.3f} (std={np.std(fold_f1s):.3f})")
    print(f"ROC-AUC:   {np.mean(fold_aucs):.3f} (std={np.std(fold_aucs):.3f})")

    cm_df = pd.DataFrame(fold_confusion_matrices)
    cm_mean = cm_df[["TN (Not Gout correct)", "FP (Gout misclassified)",
                      "FN (Gout missed)", "TP (Gout correct)"]].mean()
    print(f"\n{method_name} — 10折平均混淆矩阵:")
    print(f"  TN={cm_mean['TN (Not Gout correct)']:.1f}, FP={cm_mean['FP (Gout misclassified)']:.1f}, "
          f"FN={cm_mean['FN (Gout missed)']:.1f}, TP={cm_mean['TP (Gout correct)']:.1f}")

    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    cm_df.to_csv(f"./processed_data/cv_confusion_matrix_{safe_name}.csv", index=False)
    print(f"逐折混淆矩阵已保存到 processed_data/cv_confusion_matrix_{safe_name}.csv")

    return {
        "Accuracy_mean": np.mean(fold_accuracies), "Accuracy_std": np.std(fold_accuracies),  # <-- 新增
        "Recall_mean": np.mean(fold_recalls), "Recall_std": np.std(fold_recalls),
        "Precision_mean": np.mean(fold_precisions), "Precision_std": np.std(fold_precisions),
        "F1_mean": np.mean(fold_f1s), "F1_std": np.std(fold_f1s),
        "AUC_mean": np.mean(fold_aucs), "AUC_std": np.std(fold_aucs),
        "TN_mean": cm_mean["TN (Not Gout correct)"], "FP_mean": cm_mean["FP (Gout misclassified)"],
        "FN_mean": cm_mean["FN (Gout missed)"], "TP_mean": cm_mean["TP (Gout correct)"],
    }

# ===== 方法1: Class Weight Adjustment =====
print("\n开始跑 Class Weight Adjustment 的10折交叉验证...")
result_cw = run_cv("Class Weight Adjustment", resample_func=None)

# ===== 方法2: Undersampling (Tomek Links) =====
print("\n开始跑 Undersampling (Tomek Links) 的10折交叉验证...")
def tomek_resample(X, y):
    tomek = TomekLinks()
    return tomek.fit_resample(X, y)

result_tomek = run_cv("Undersampling (Tomek Links)", resample_func=tomek_resample)

# ===== 汇总保存 =====
summary = pd.DataFrame({
    "Class Weight Adjustment": result_cw,
    "Undersampling (Tomek Links)": result_tomek,
}).T

print("\n\n========== 10折交叉验证 最终汇总 ==========")
print(summary)
summary.to_csv("./processed_data/cv_comparison.csv")
print("\n结果已保存到 processed_data/cv_comparison.csv")
print("(该汇总表已包含Accuracy、TN/FP/FN/TP平均值列；逐折混淆矩阵详见各方法对应的 cv_confusion_matrix_*.csv 文件)")