import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# ===== 读取数据 =====
X_train = np.load("./processed_data/train_embeddings.npy")
X_test = np.load("./processed_data/test_embeddings.npy")
y_train = pd.read_csv("./processed_data/train_labels.csv")["Predict"]
y_test = pd.read_csv("./processed_data/test_labels.csv")["Predict"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_test_binary = (y_test == "Y").astype(int)

results = {}

def evaluate(name, y_pred, y_score=None):
    recall = recall_score(y_test, y_pred, pos_label="Y")
    f1 = f1_score(y_test, y_pred, pos_label="Y")
    auc = roc_auc_score(y_test_binary, y_score) if y_score is not None else None
    results[name] = {"Recall": recall, "F1": f1, "ROC-AUC": auc}
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["Not Gout", "Gout"], labels=["N", "Y"]))
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

# ===== 方法1: Class Weight Adjustment（baseline，已跑过，这里重新跑一次方便统一比较）=====
print("训练方法1: Class Weight Adjustment...")
svm_cw = SVC(kernel="linear", class_weight="balanced", random_state=42)
svm_cw.fit(X_train_scaled, y_train)
y_pred_cw = svm_cw.predict(X_test_scaled)
y_score_cw = svm_cw.decision_function(X_test_scaled)  # 用decision_function代替predict_proba，速度快很多
evaluate("Class Weight Adjustment", y_pred_cw, y_score_cw)

# ===== 方法2: Undersampling (Tomek Links) =====
print("\n训练方法2: Undersampling (Tomek Links)...")
tomek = TomekLinks()
X_train_tomek, y_train_tomek = tomek.fit_resample(X_train_scaled, y_train)
print(f"Tomek Links处理后训练集大小: {X_train_tomek.shape}, 标签分布: {pd.Series(y_train_tomek).value_counts().to_dict()}")

svm_tomek = SVC(kernel="linear", random_state=42)  # 已经欠采样平衡过，不需要再class_weight
svm_tomek.fit(X_train_tomek, y_train_tomek)
y_pred_tomek = svm_tomek.predict(X_test_scaled)
y_score_tomek = svm_tomek.decision_function(X_test_scaled)
evaluate("Undersampling (Tomek Links)", y_pred_tomek, y_score_tomek)

# ===== 方法3: Ensemble (BalancedBagging) =====
print("\n训练方法3: Ensemble (BalancedBagging)...")
balanced_bagging = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    sampling_strategy="auto",
    replacement=False,
    random_state=42
)
balanced_bagging.fit(X_train_scaled, y_train)
y_pred_bb = balanced_bagging.predict(X_test_scaled)
y_score_bb = balanced_bagging.predict_proba(X_test_scaled)[:, list(balanced_bagging.classes_).index("Y")]
evaluate("Ensemble (BalancedBagging)", y_pred_bb, y_score_bb)

# ===== 方法4: Threshold Moving（基于方法1的SVM，调整决策阈值）=====
print("\n训练方法4: Threshold Moving（基于Class Weight SVM调整阈值）...")
# 默认阈值是0，尝试把阈值降低，让模型更容易预测为Gout（提高recall）
threshold = -0.5  # 可以后续调整这个数字来找最佳平衡点
y_pred_threshold = np.where(y_score_cw > threshold, "Y", "N")
evaluate(f"Threshold Moving (threshold={threshold})", y_pred_threshold, y_score_cw)

# ===== 汇总对比 =====
print("\n\n========== 四种方法汇总对比 ==========")
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv("./processed_data/imbalance_methods_comparison.csv")
print("\n结果已保存到 processed_data/imbalance_methods_comparison.csv")