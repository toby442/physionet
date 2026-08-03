import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ===== 加载之前已经下载好的本地BlueBERT模型 =====
local_dir = "./bluebert_local"
print("正在加载BlueBERT模型...")
tokenizer = BertTokenizer.from_pretrained(local_dir)
model = BertModel.from_pretrained(local_dir)
model.eval()  # 设置为推理模式（不训练，只提取特征）

# ===== 读取之前处理好的训练集/测试集 =====
train_df = pd.read_csv("./processed_data/train.csv")
test_df = pd.read_csv("./processed_data/test.csv")

print(f"训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")

def get_cls_embeddings(texts, batch_size=16):
    """
    把一批文本转换成BlueBERT的[CLS] embedding向量
    每次处理batch_size条，避免内存占用过大
    """
    all_embeddings = []
    texts = [str(t) for t in texts]  # 防止有缺失值导致报错

    for i in tqdm(range(0, len(texts), batch_size), desc="提取向量中"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # 大部分chief complaint都很短，128足够
        )

        with torch.no_grad():  # 不计算梯度，节省内存和时间
            outputs = model(**inputs)

        # 取每条文本的[CLS] token向量（第0个token的输出）
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

# ===== 提取训练集向量 =====
print("\n开始提取训练集向量...")
train_embeddings = get_cls_embeddings(train_df["Chief Complaint"].tolist())
print(f"训练集向量维度: {train_embeddings.shape}")

# ===== 提取测试集向量 =====
print("\n开始提取测试集向量...")
test_embeddings = get_cls_embeddings(test_df["Chief Complaint"].tolist())
print(f"测试集向量维度: {test_embeddings.shape}")

# ===== 保存向量和对应标签，方便后续直接读取，不用重复计算 =====
output_dir = "./processed_data"
np.save(os.path.join(output_dir, "train_embeddings.npy"), train_embeddings)
np.save(os.path.join(output_dir, "test_embeddings.npy"), test_embeddings)
train_df["Predict"].to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
test_df["Predict"].to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

print("\n向量提取完成，已保存到 processed_data 文件夹！")