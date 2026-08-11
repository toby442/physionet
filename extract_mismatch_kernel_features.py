"""
Mismatch Kernel 特征提取脚本
结构与 extract_string_kernel_features.py 完全一致，只是把
CountVectorizer(analyzer='char', ngram_range=(3,5)) 换成了容忍拼写变体的
MismatchVectorizer(ngram_range=(3,5), m=1)。

输出文件命名和格式与现有 String Kernel 完全一致，直接对应到
cross_validation.py 里 feature_sets 列表加一行即可接入现有8组合框架，
不需要改动 cross_validation.py 的任何其他代码。

运行方式：跟 extract_string_kernel_features.py 放在同一个目录下直接运行即可：
    python extract_mismatch_kernel_features.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import os
import time


class MismatchVectorizer:
    """
    支持多长度混合(ngram_range)的显式 Mismatch(k,m) Kernel 特征提取器。
    用法与 CountVectorizer 一致：fit_transform(texts) -> 特征矩阵。

    m=0 时数学上等价于 CountVectorizer(analyzer='char', ngram_range=ngram_range)，
    已验证在相同词表下逐元素结果完全一致，因此与现有 Spectrum kernel 基线具备
    严格可比性（唯一变量是是否容忍字符误差）。

    参数:
        ngram_range: 与 CountVectorizer 含义一致，(3,5) 表示同时考虑
                     3/4/5字符长度的片段，混合后按频率取前 max_features 个
        m: 允许的最大字符不匹配数(Hamming距离)。m=1 是最常用的起点
        max_features: 词表大小上限，与现有 Spectrum kernel 保持一致(5000)
                      以保证两者是公平对比(只改变匹配严格度，不改变维度)
    """

    def __init__(self, ngram_range=(3, 5), m=1, max_features=5000):
        self.ngram_range = ngram_range
        self.m = m
        self.max_features = max_features
        self.vocabulary_ = None
        self._vocab_by_len = None        # {length: encoded_array (V_len, length)}
        self._vocab_index_by_len = None  # {length: [位置索引，对应全局vocabulary_里的顺序]}

    def _build_vocab(self, texts):
        # 用CountVectorizer统计高频片段作为词表，逻辑与现有Spectrum kernel完全一致
        cv = CountVectorizer(analyzer='char', ngram_range=self.ngram_range,
                              max_features=self.max_features, lowercase=True)
        cv.fit(texts)
        self.vocabulary_ = cv.get_feature_names_out().tolist()

        # 按长度分组（Hamming距离只能在等长字符串之间计算）
        self._vocab_by_len = {}
        self._vocab_index_by_len = {}
        for idx, term in enumerate(self.vocabulary_):
            L = len(term)
            self._vocab_index_by_len.setdefault(L, []).append(idx)
            self._vocab_by_len.setdefault(L, []).append([ord(c) for c in term])
        for L in self._vocab_by_len:
            self._vocab_by_len[L] = np.array(self._vocab_by_len[L], dtype=np.int32)

    def _extract_windows(self, text, L):
        text = text.lower()
        n = len(text)
        if n < L:
            return np.empty((0, L), dtype=np.int32)
        return np.array(
            [[ord(c) for c in text[i:i + L]] for i in range(n - L + 1)],
            dtype=np.int32
        )

    def _transform_one(self, text):
        vec = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for L, vocab_arr in self._vocab_by_len.items():
            windows = self._extract_windows(text, L)
            if windows.shape[0] == 0:
                continue
            # 广播计算Hamming距离: (n_windows, V_len, L) -> sum over L -> (n_windows, V_len)
            diffs = (windows[:, None, :] != vocab_arr[None, :, :])
            dist = diffs.sum(axis=2)
            hits = (dist <= self.m).sum(axis=0)  # (V_len,)
            positions = self._vocab_index_by_len[L]
            vec[positions] = hits
        return vec

    def fit(self, texts):
        self._build_vocab(texts)
        return self

    def transform(self, texts):
        rows = [self._transform_one(t) for t in texts]
        return np.vstack(rows)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# ===================== 主流程（与 extract_string_kernel_features.py 完全对应）=====================

# ===== 读取redacted版本的完整清洗数据集（与现有脚本读取同一份数据）=====
clean_df = pd.read_csv("./processed_data/clean_full_redacted.csv")
print(f"数据集大小: {len(clean_df)}")
print(f"标签分布:\n{clean_df['Predict'].value_counts()}")

texts = clean_df["Chief Complaint"].astype(str).tolist()
labels = clean_df["Predict"]

# ===== 与现有 Spectrum kernel 保持一致的 ngram_range 和 max_features，
#       唯一改变的变量是 m（是否容忍字符误差）=====
vectorizer = MismatchVectorizer(
    ngram_range=(3, 5),
    m=1,
    max_features=5000,
)

print("\n正在提取 mismatch kernel（容忍拼写变体的字符k-gram）特征...")
start = time.time()
X_mismatch_dense = vectorizer.fit_transform(texts)
elapsed = time.time() - start
print(f"特征矩阵维度: {X_mismatch_dense.shape}，耗时: {elapsed:.1f}秒")

# ===== 转成稀疏矩阵，保持与现有 string_kernel_features_redacted.npz 完全一致的存储格式 =====
X_mismatch = sparse.csr_matrix(X_mismatch_dense)

# ===== 输出文件名对应到 mismatch_kernel 前缀，与现有 string_kernel 命名习惯一致 =====
output_dir = "./processed_data"
sparse.save_npz(os.path.join(output_dir, "mismatch_kernel_features_redacted.npz"), X_mismatch)
labels.to_csv(os.path.join(output_dir, "mismatch_kernel_labels_redacted.csv"), index=False)

print("\nmismatch kernel特征提取完成，已保存！")
print(f"举例：前10个词表片段：{vectorizer.vocabulary_[:10]}")