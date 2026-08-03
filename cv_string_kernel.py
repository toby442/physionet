import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# ===== 读取string kernel特征 =====
X_full = sparse.load_npz("./processed_data/string_kernel_features.npz")
y_full = pd.read_csv("./processed_data/string_kernel_labels.csv")["Predict"]

print(f"String Kernel特征矩阵: {X_full.shape}")
print(f"标签分布: {y_full.value_counts().to_dict()}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def summarize(name, recalls, precisions, f1s, aucs):
    print(f"\n===== {name} — 10折平均结果 =====")
    print(f"Recall:    {np.mean(recalls):.3f} (std={np.std(recalls):.3f})")
    print(f"Precision: {np.mean(precisions):.3f} (std={np.std(precisions):.3f})")
    print(f"F1:        {np.mean(f1s):.3f} (std={np.std(f1s):.3f})")
    print(f"ROC-AUC:   {np.mean(aucs):.3f} (std={np.std(aucs):.3f})")
    return {
        "Recall_mean": np.mean(recalls), "Recall_std": np.std(recalls),
        "Precision_mean": np.mean(precisions), "Precision_std": np.std(precisions),
        "F1_mean": np.mean(f1s), "F1_std": np.std(f1s),
        "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs),
    }

def run_cv(name, resample_func=None, use_class_weight=False):
    recalls, precisions, f1s, aucs = [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        # 注意：string kernel特征是稀疏矩阵，StandardScaler需要设置with_mean=False才能处理稀疏数据
        scaler = StandardScaler(with_mean=False)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if resample_func is not None:
            X_tr_scaled, y_tr = resample_func(X_tr_scaled, y_tr)

        if use_class_weight:
            svm = SVC(kernel="linear", class_weight="balanced", random_state=42)
        else:
            svm = SVC(kernel="linear", random_state=42)

        svm.fit(X_tr_scaled, y_tr)
        y_pred = svm.predict(X_val_scaled)
        y_score = svm.decision_function(X_val_scaled)
        y_val_binary = (y_val == "Y").astype(int)

        recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
        precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
        auc = roc_auc_score(y_val_binary, y_score)

        recalls.append(recall); precisions.append(precision)
        f1s.append(f1); aucs.append(auc)
        print(f"  Fold {fold_idx}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

    return summarize(name, recalls, precisions, f1s, aucs)

# ===== 方法1: Class Weight Adjustment =====
print("\n开始跑 String Kernel + Class Weight Adjustment...")
result_cw = run_cv("String Kernel + Class Weight", resample_func=None, use_class_weight=True)

# ===== 方法2: Undersampling (Tomek Links) =====
print("\n开始跑 String Kernel + Undersampling (Tomek Links)...")
def tomek_resample(X, y):
    tomek = TomekLinks()
    return tomek.fit_resample(X, y)
result_tomek = run_cv("String Kernel + Tomek Links", resample_func=tomek_resample, use_class_weight=False)

# ===== 汇总保存 =====
summary = pd.DataFrame({
    "String Kernel + Class Weight": result_cw,
    "String Kernel + Tomek Links": result_tomek,
}).T

print("\n\n========== String Kernel 方法汇总 ==========")
print(summary)
summary.to_csv("./processed_data/cv_string_kernel_comparison.csv")
print("\n结果已保存到 processed_data/cv_string_kernel_comparison.csv")