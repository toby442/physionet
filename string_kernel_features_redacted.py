import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import os

# ===== 改动点：读取redacted版本的完整清洗数据集 =====
clean_df = pd.read_csv("./processed_data/clean_full_redacted.csv")
print(f"数据集大小: {len(clean_df)}")
print(f"标签分布:\n{clean_df['Predict'].value_counts()}")

texts = clean_df["Chief Complaint"].astype(str).tolist()
labels = clean_df["Predict"]

vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000,
    lowercase=True
)

print("\n正在提取string kernel（字符k-gram）特征...")
X_string_kernel = vectorizer.fit_transform(texts)
print(f"特征矩阵维度: {X_string_kernel.shape}")

# ===== 改动点：所有输出文件名加 _redacted 后缀 =====
output_dir = "./processed_data"
sparse.save_npz(os.path.join(output_dir, "string_kernel_features_redacted.npz"), X_string_kernel)
labels.to_csv(os.path.join(output_dir, "string_kernel_labels_redacted.csv"), index=False)

print("\nstring kernel特征提取完成，已保存（文件名带_redacted后缀）！")
print(f"举例：前10个最常见的字符片段：{vectorizer.get_feature_names_out()[:10]}")