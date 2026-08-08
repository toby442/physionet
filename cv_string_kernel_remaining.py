import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, accuracy_score
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# ===== 读取string kernel特征 =====
X_full = sparse.load_npz("./processed_data/string_kernel_features.npz")
y_full = pd.read_csv("./processed_data/string_kernel_labels.csv")["Predict"]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def summarize(name, accuracies, recalls, precisions, f1s, aucs):
    print(f"\n===== {name} — 10折平均结果 =====")
    print(f"Accuracy:  {np.mean(accuracies):.3f} (std={np.std(accuracies):.3f})")
    print(f"Recall:    {np.mean(recalls):.3f} (std={np.std(recalls):.3f})")
    print(f"Precision: {np.mean(precisions):.3f} (std={np.std(precisions):.3f})")
    print(f"F1:        {np.mean(f1s):.3f} (std={np.std(f1s):.3f})")
    print(f"ROC-AUC:   {np.mean(aucs):.3f} (std={np.std(aucs):.3f})")
    return {
        "Accuracy_mean": np.mean(accuracies), "Accuracy_std": np.std(accuracies),
        "Recall_mean": np.mean(recalls), "Recall_std": np.std(recalls),
        "Precision_mean": np.mean(precisions), "Precision_std": np.std(precisions),
        "F1_mean": np.mean(f1s), "F1_std": np.std(f1s),
        "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs),
    }

# ===== 方法3: Ensemble (BalancedBagging) =====
print("开始跑 String Kernel + BalancedBagging 的10折交叉验证...")
bb_accuracies, bb_recalls, bb_precisions, bb_f1s, bb_aucs = [], [], [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
    X_tr, X_val = X_full[train_idx], X_full[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

    scaler = StandardScaler(with_mean=False)
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    bb = BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(),
        sampling_strategy="auto",
        replacement=False,
        random_state=42
    )
    bb.fit(X_tr_scaled, y_tr)
    y_pred = bb.predict(X_val_scaled)
    y_score = bb.predict_proba(X_val_scaled)[:, list(bb.classes_).index("Y")]
    y_val_binary = (y_val == "Y").astype(int)

    recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
    f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
    precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
    auc = roc_auc_score(y_val_binary, y_score)
    accuracy = accuracy_score(y_val, y_pred)

    bb_accuracies.append(accuracy)
    bb_recalls.append(recall); bb_precisions.append(precision)
    bb_f1s.append(f1); bb_aucs.append(auc)
    print(f"  Fold {fold_idx}: Accuracy={accuracy:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

result_bb = summarize("String Kernel + BalancedBagging", bb_accuracies, bb_recalls, bb_precisions, bb_f1s, bb_aucs)

# ===== 方法4: Threshold Moving =====
print("\n开始跑 String Kernel + Threshold Moving 的10折交叉验证...")
tm_accuracies, tm_recalls, tm_precisions, tm_f1s, tm_aucs = [], [], [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
    X_tr, X_val = X_full[train_idx], X_full[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

    scaler = StandardScaler(with_mean=False)
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    svm_for_threshold = SVC(kernel="linear", class_weight="balanced", random_state=42)
    inner_scores = cross_val_predict(
        svm_for_threshold, X_tr_scaled, y_tr, cv=3, method="decision_function"
    )

    candidate_thresholds = np.linspace(-2, 2, 41)
    best_threshold, best_f1 = 0, -1
    for t in candidate_thresholds:
        pred_temp = np.where(inner_scores > t, "Y", "N")
        f1_temp = f1_score(y_tr, pred_temp, pos_label="Y", zero_division=0)
        if f1_temp > best_f1:
            best_f1, best_threshold = f1_temp, t

    final_svm = SVC(kernel="linear", class_weight="balanced", random_state=42)
    final_svm.fit(X_tr_scaled, y_tr)
    y_score = final_svm.decision_function(X_val_scaled)
    y_pred = np.where(y_score > best_threshold, "Y", "N")
    y_val_binary = (y_val == "Y").astype(int)

    recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
    f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
    precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
    auc = roc_auc_score(y_val_binary, y_score)
    accuracy = accuracy_score(y_val, y_pred)

    tm_accuracies.append(accuracy)
    tm_recalls.append(recall); tm_precisions.append(precision)
    tm_f1s.append(f1); tm_aucs.append(auc)
    print(f"  Fold {fold_idx}: best_threshold={best_threshold:.2f}, Accuracy={accuracy:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

result_tm = summarize("String Kernel + Threshold Moving", tm_accuracies, tm_recalls, tm_precisions, tm_f1s, tm_aucs)

# ===== 合并全部结果:8种组合(BlueBERT 4种 + String Kernel 4种) =====
try:
    bluebert_results = pd.read_csv("./processed_data/cv_comparison_full.csv", index_col=0)
except FileNotFoundError:
    bluebert_results = pd.DataFrame()

try:
    stringkernel_cw_tomek = pd.read_csv("./processed_data/cv_string_kernel_comparison.csv", index_col=0)
except FileNotFoundError:
    stringkernel_cw_tomek = pd.DataFrame()

new_results = pd.DataFrame({
    "String Kernel + BalancedBagging": result_bb,
    "String Kernel + Threshold Moving": result_tm,
}).T

full_summary = pd.concat([bluebert_results, stringkernel_cw_tomek, new_results])

print("\n\n========== 全部8种组合 10折交叉验证 完整汇总 ==========")
print(full_summary)
full_summary.to_csv("./processed_data/cv_all_8_methods_comparison.csv")
print("\n结果已保存到 processed_data/cv_all_8_methods_comparison.csv")