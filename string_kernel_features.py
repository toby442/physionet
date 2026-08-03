import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import os

# ===== 读取之前清洗好的完整数据集(不区分train/test，先统一提取特征) =====
clean_df = pd.read_csv("./processed_data/clean_full.csv")
print(f"数据集大小: {len(clean_df)}")
print(f"标签分布:\n{clean_df['Predict'].value_counts()}")

texts = clean_df["Chief Complaint"].astype(str).tolist()
labels = clean_df["Predict"]

# ===== 用字符k-gram向量化，近似实现Spectrum String Kernel =====
# analyzer='char' 表示按字符切分（不是按单词），ngram_range=(3,5) 表示同时统计3、4、5个连续字符的组合
# 这样能捕捉到医学缩写、拼写变体等字符级别的模式（比如"c/o"、"pt"这类缩写）
vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000,   # 只保留最常见的5000个字符片段，控制特征维度，避免过于稀疏
    lowercase=True
)

print("\n正在提取string kernel（字符k-gram）特征...")
X_string_kernel = vectorizer.fit_transform(texts)
print(f"特征矩阵维度: {X_string_kernel.shape}")  # (样本数, 特征数)

# ===== 保存特征矩阵和标签，方便后续直接读取 =====
output_dir = "./processed_data"
sparse.save_npz(os.path.join(output_dir, "string_kernel_features.npz"), X_string_kernel)
labels.to_csv(os.path.join(output_dir, "string_kernel_labels.csv"), index=False)

print("\nstring kernel特征提取完成，已保存！")
print(f"举例：前10个最常见的字符片段：{vectorizer.get_feature_names_out()[:10]}")