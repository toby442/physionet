import pandas as pd

output_dir = "./processed_data"

synthetic_df = pd.read_csv(f"{output_dir}/clean_full.csv")
redacted_df = pd.read_csv(f"{output_dir}/clean_full_redacted.csv")

# 找出文本不同的记录
diff_mask = synthetic_df["Chief Complaint"].astype(str) != redacted_df["Chief Complaint"].astype(str)
diff_indices = synthetic_df[diff_mask].index

print(f"共 {len(diff_indices)} 条记录文本不同\n")
print("===== 前20条具体对比 =====\n")

for i, idx in enumerate(diff_indices[:20]):
    label = synthetic_df.loc[idx, "Predict"]
    syn_text = synthetic_df.loc[idx, "Chief Complaint"]
    red_text = redacted_df.loc[idx, "Chief Complaint"]
    print(f"[{i+1}] 标签={label}")
    print(f"  Synthetic: {syn_text}")
    print(f"  Redacted : {red_text}")
    print()

# 统计一下差异是否集中在某些特定模式上（比如长度差异、特定占位符）
print("\n===== 差异模式统计 =====")
length_diffs = []
for idx in diff_indices:
    syn_text = str(synthetic_df.loc[idx, "Chief Complaint"])
    red_text = str(redacted_df.loc[idx, "Chief Complaint"])
    length_diffs.append(len(red_text) - len(syn_text))

print(f"Redacted比Synthetic平均长度差: {sum(length_diffs)/len(length_diffs):.1f} 字符")
print(f"长度差异范围: 最小{min(length_diffs)}, 最大{max(length_diffs)}")

# 看这80条里，Y和N标签各占多少，判断差异是否偏向某一类
diff_label_dist = synthetic_df.loc[diff_indices, "Predict"].value_counts()
print(f"\n差异记录的标签分布：\n{diff_label_dist}")