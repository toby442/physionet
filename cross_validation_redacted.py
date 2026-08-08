import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, precision_score,
    confusion_matrix, accuracy_score,
)
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

SVC_MAX_ITER = 100000
data_dir = "./processed_data"

# ===== 1. 读取两种特征：BlueBERT（合并train+test）和String Kernel（本来就是全量） =====
train_emb = np.load(f"{data_dir}/train_embeddings_redacted.npy")
test_emb = np.load(f"{data_dir}/test_embeddings_redacted.npy")
train_labels = pd.read_csv(f"{data_dir}/train_labels_redacted.csv")["Predict"]
test_labels = pd.read_csv(f"{data_dir}/test_labels_redacted.csv")["Predict"]

X_bluebert = np.vstack([train_emb, test_emb])
y_bluebert = pd.concat([train_labels, test_labels], ignore_index=True)

X_stringkernel = sparse.load_npz(f"{data_dir}/string_kernel_features_redacted.npz")
y_stringkernel = pd.read_csv(f"{data_dir}/string_kernel_labels_redacted.csv")["Predict"]

print(f"BlueBERT特征: {X_bluebert.shape}, 标签分布: {y_bluebert.value_counts().to_dict()}")
print(f"String Kernel特征: {X_stringkernel.shape}, 标签分布: {y_stringkernel.value_counts().to_dict()}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def load_progress(key):
    path = f"{data_dir}/cv_progress_redacted_{key}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"[检测到{key}已有进度，已完成{len(data)}折，将从第{len(data)+1}折继续]")
        return data
    return []


def save_progress(key, fold_results):
    with open(f"{data_dir}/cv_progress_redacted_{key}.json", "w") as f:
        json.dump(fold_results, f, indent=2)


def save_result(key, result_dict):
    with open(f"{data_dir}/cv_result_redacted_{key}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"[已存档最终结果] cv_result_redacted_{key}.json")


def already_done(key):
    path = f"{data_dir}/cv_result_redacted_{key}.json"
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        print(f"[检测到已有完整结果，跳过重跑] {key}")
        return result
    return None


def aggregate(fold_results):
    df = pd.DataFrame(fold_results)
    return {
        "Accuracy_mean": float(df["Accuracy"].mean()), "Accuracy_std": float(df["Accuracy"].std()),
        "Recall_mean": float(df["Recall"].mean()), "Recall_std": float(df["Recall"].std()),
        "Precision_mean": float(df["Precision"].mean()), "Precision_std": float(df["Precision"].std()),
        "F1_mean": float(df["F1"].mean()), "F1_std": float(df["F1"].std()),
        "AUC_mean": float(df["AUC"].mean()), "AUC_std": float(df["AUC"].std()),
    }


def run_cv(feature_name, X_full, y_full, method_name, key, resample_func=None,
           use_ensemble=False, is_sparse=False):
    fold_results = load_progress(key)
    done_folds = {r["Fold"] for r in fold_results}
    fold_list = list(skf.split(X_full, y_full))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_list, 1):
        if fold_idx in done_folds:
            continue

        fold_start = time.time()
        if is_sparse:
            X_tr, X_val = X_full[train_idx], X_full[val_idx]
        else:
            X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        scaler = StandardScaler(with_mean=False) if is_sparse else StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if resample_func is not None:
            X_tr_scaled, y_tr = resample_func(X_tr_scaled, y_tr)

        if use_ensemble:
            model = BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(),
                sampling_strategy="auto", replacement=False,
                random_state=42, n_jobs=1,
            )
            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_val_scaled)
            y_score = model.predict_proba(X_val_scaled)[:, list(model.classes_).index("Y")]
        else:
            if method_name == "Class Weight Adjustment":
                model = SVC(kernel="linear", class_weight="balanced", random_state=42, max_iter=SVC_MAX_ITER)
            else:
                model = SVC(kernel="linear", random_state=42, max_iter=SVC_MAX_ITER)
            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_val_scaled)
            y_score = model.decision_function(X_val_scaled)

        y_val_binary = (y_val == "Y").astype(int)
        recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
        precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
        auc = roc_auc_score(y_val_binary, y_score)
        accuracy = accuracy_score(y_val, y_pred)

        fold_time = time.time() - fold_start
        print(f"  [{feature_name} + {method_name}] Fold {fold_idx}: Accuracy={accuracy:.4f}, "
              f"Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}, AUC={auc:.4f}, "
              f"耗时={fold_time:.1f}秒")

        fold_results.append({
            "Fold": fold_idx, "Accuracy": accuracy, "Recall": recall,
            "Precision": precision, "F1": f1, "AUC": auc,
        })
        save_progress(key, fold_results)

    result = aggregate(fold_results)
    print(f"\n===== {feature_name} + {method_name} — 10折平均结果 =====")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    return result


def run_threshold_moving(feature_name, X_full, y_full, key, is_sparse=False):
    fold_results = load_progress(key)
    done_folds = {r["Fold"] for r in fold_results}
    fold_list = list(skf.split(X_full, y_full))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_list, 1):
        if fold_idx in done_folds:
            continue

        fold_start = time.time()
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        scaler = StandardScaler(with_mean=False) if is_sparse else StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        svm_for_threshold = SVC(kernel="linear", class_weight="balanced", random_state=42, max_iter=SVC_MAX_ITER)
        inner_scores = cross_val_predict(
            svm_for_threshold, X_tr_scaled, y_tr, cv=3, method="decision_function", n_jobs=1
        )
        y_tr_binary = (y_tr == "Y").astype(int).values

        candidate_thresholds = np.linspace(-2, 2, 41)
        best_threshold, best_f1 = 0, -1
        for t in candidate_thresholds:
            pred_temp = np.where(inner_scores > t, 1, 0)
            f1_temp = f1_score(y_tr_binary, pred_temp, pos_label=1, zero_division=0)
            if f1_temp > best_f1:
                best_f1, best_threshold = f1_temp, t

        final_svm = SVC(kernel="linear", class_weight="balanced", random_state=42, max_iter=SVC_MAX_ITER)
        final_svm.fit(X_tr_scaled, y_tr)
        y_score = final_svm.decision_function(X_val_scaled)
        y_pred_binary = np.where(y_score > best_threshold, 1, 0)
        y_pred = np.where(y_pred_binary == 1, "Y", "N")

        y_val_binary = (y_val == "Y").astype(int)
        recall = recall_score(y_val, y_pred, pos_label="Y", zero_division=0)
        precision = precision_score(y_val, y_pred, pos_label="Y", zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label="Y", zero_division=0)
        auc = roc_auc_score(y_val_binary, y_score)
        accuracy = accuracy_score(y_val, y_pred)

        fold_time = time.time() - fold_start
        print(f"  [{feature_name} + Threshold Moving] Fold {fold_idx}: best_threshold={best_threshold:.2f}, "
              f"Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}, "
              f"AUC={auc:.4f}, 耗时={fold_time:.1f}秒")

        fold_results.append({
            "Fold": fold_idx, "Accuracy": accuracy, "Recall": recall,
            "Precision": precision, "F1": f1, "AUC": auc,
        })
        save_progress(key, fold_results)

    result = aggregate(fold_results)
    print(f"\n===== {feature_name} + Threshold Moving — 10折平均结果 =====")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    return result


def tomek_resample(X, y):
    tomek = TomekLinks(n_jobs=1)
    return tomek.fit_resample(X, y)


overall_start = time.time()
all_results = {}

feature_sets = [
    ("BlueBERT", X_bluebert, y_bluebert, False),
    ("String Kernel", X_stringkernel, y_stringkernel, True),
]

for feature_name, X_full, y_full, is_sparse in feature_sets:
    feature_key = "bluebert" if feature_name == "BlueBERT" else "stringkernel"

    method_configs = [
        ("Class Weight Adjustment", f"{feature_key}_class_weight", None, False),
        ("Undersampling (Tomek Links)", f"{feature_key}_tomek", tomek_resample, False),
        ("Ensemble (BalancedBagging)", f"{feature_key}_balanced_bagging", None, True),
    ]

    for method_name, key, resample_func, use_ensemble in method_configs:
        existing = already_done(key)
        combo_name = f"{feature_name} + {method_name}"
        if existing:
            all_results[combo_name] = existing
            continue
        print("\n" + "=" * 20 + f" 开始跑 {combo_name} " + "=" * 20)
        result = run_cv(feature_name, X_full, y_full, method_name, key,
                         resample_func=resample_func, use_ensemble=use_ensemble, is_sparse=is_sparse)
        save_result(key, result)
        all_results[combo_name] = result

    key = f"{feature_key}_threshold_moving"
    combo_name = f"{feature_name} + Threshold Moving"
    existing = already_done(key)
    if existing:
        all_results[combo_name] = existing
    else:
        print("\n" + "=" * 20 + f" 开始跑 {combo_name} " + "=" * 20)
        result = run_threshold_moving(feature_name, X_full, y_full, key, is_sparse=is_sparse)
        save_result(key, result)
        all_results[combo_name] = result

summary = pd.DataFrame(all_results).T
print("\n\n========== Redacted版本 8种组合 10折交叉验证 最终汇总 ==========")
print(summary)
summary.to_csv(f"{data_dir}/cv_all_8_methods_comparison_redacted.csv")

total_time = time.time() - overall_start
print(f"\n结果已保存到 {data_dir}/cv_all_8_methods_comparison_redacted.csv")
print(f"本次运行耗时: {total_time/60:.1f} 分钟（已跳过之前完成的方法/折）")