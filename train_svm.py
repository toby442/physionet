import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===== 读取之前提取好的BlueBERT向量和标签 =====
X_train = np.load("./processed_data/train_embeddings.npy")
X_test = np.load("./processed_data/test_embeddings.npy")
y_train = pd.read_csv("./processed_data/train_labels.csv")["Predict"]
y_test = pd.read_csv("./processed_data/test_labels.csv")["Predict"]

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ===== 标准化特征（SVM对特征尺度敏感，proposal里也提到用StandardScaler）=====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 训练SVM，使用class_weight='balanced'处理类别不平衡 =====
print("\n开始训练SVM（class_weight=balanced）...")
svm = SVC(
    kernel="linear",       # proposal里用的linear kernel
    class_weight="balanced",  # 自动根据类别频率反比加权
    probability=True,      # 允许输出概率，方便算ROC-AUC
    random_state=42
)
svm.fit(X_train_scaled, y_train)
print("训练完成！")

# ===== 在测试集上评估 =====
y_pred = svm.predict(X_test_scaled)

print("\n===== 分类报告 (Classification Report) =====")
print(classification_report(y_test, y_pred, target_names=["Not Gout (N)", "Gout (Y)"], labels=["N", "Y"]))

print("\n===== 混淆矩阵 (Confusion Matrix) =====")
cm = confusion_matrix(y_test, y_pred, labels=["N", "Y"])
print("            Predicted N   Predicted Y")
print(f"Actual N    {cm[0][0]:<13}{cm[0][1]}")
print(f"Actual Y    {cm[1][0]:<13}{cm[1][1]}")

# ===== ROC-AUC（把标签转成0/1数值，Y=1为正类）=====
y_test_binary = (y_test == "Y").astype(int)
y_proba = svm.predict_proba(X_test_scaled)[:, list(svm.classes_).index("Y")]
auc = roc_auc_score(y_test_binary, y_proba)
print(f"\nROC-AUC: {auc:.4f}")