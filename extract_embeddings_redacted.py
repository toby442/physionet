import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ===== 加载之前已经下载好的本地BlueBERT模型（和synthetic版本用的是同一个模型文件，不用重新下载） =====
local_dir = "./bluebert_local"
print("正在加载BlueBERT模型...")
tokenizer = BertTokenizer.from_pretrained(local_dir)
model = BertModel.from_pretrained(local_dir)
model.eval()

# ===== 改动点：读取redacted版本的训练集/测试集 =====
train_df = pd.read_csv("./processed_data/train_redacted.csv")
test_df = pd.read_csv("./processed_data/test_redacted.csv")

print(f"训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")


def get_cls_embeddings(texts, batch_size=16):
    all_embeddings = []
    texts = [str(t) for t in texts]

    for i in tqdm(range(0, len(texts), batch_size), desc="提取向量中"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


print("\n开始提取训练集向量...")
train_embeddings = get_cls_embeddings(train_df["Chief Complaint"].tolist())
print(f"训练集向量维度: {train_embeddings.shape}")

print("\n开始提取测试集向量...")
test_embeddings = get_cls_embeddings(test_df["Chief Complaint"].tolist())
print(f"测试集向量维度: {test_embeddings.shape}")

# ===== 改动点：所有输出文件名加 _redacted 后缀，不覆盖synthetic版本已有的结果 =====
output_dir = "./processed_data"
np.save(os.path.join(output_dir, "train_embeddings_redacted.npy"), train_embeddings)
np.save(os.path.join(output_dir, "test_embeddings_redacted.npy"), test_embeddings)
train_df["Predict"].to_csv(os.path.join(output_dir, "train_labels_redacted.csv"), index=False)
test_df["Predict"].to_csv(os.path.join(output_dir, "test_labels_redacted.csv"), index=False)

print("\n向量提取完成，已保存到 processed_data 文件夹（文件名带_redacted后缀）！")