from huggingface_hub import hf_hub_download
import json, os
from transformers import BertTokenizer, BertModel

model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
local_dir = "./bluebert_local"
os.makedirs(local_dir, exist_ok=True)

# 只下载真正需要的3个文件，跳过无用的大文件
files_needed = ["config.json", "vocab.txt", "pytorch_model.bin"]

for f in files_needed:
    path = hf_hub_download(repo_id=model_name, filename=f, local_dir=local_dir)
    print(f"下载完成: {f}")

# 检查config.json里是否有model_type，没有的话手动补上（BlueBERT本质是bert结构）
config_path = os.path.join(local_dir, "config.json")
with open(config_path, "r") as file:
    config_data = json.load(file)

if "model_type" not in config_data:
    config_data["model_type"] = "bert"
    with open(config_path, "w") as file:
        json.dump(config_data, file, indent=2)
    print("已修复 config.json，补上 model_type 字段")

print("正在从本地加载模型...")
tokenizer = BertTokenizer.from_pretrained(local_dir)
model = BertModel.from_pretrained(local_dir)

print("模型加载成功！")

text = "pt c/o sudden pain and swelling big toe overnight"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print("句子编码成功，输出向量维度：", outputs.last_hidden_state.shape)