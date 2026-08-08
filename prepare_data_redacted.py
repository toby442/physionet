import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_dir = r"D:\硕士毕业课题\dataset\sorted"

# ===== 改动点1:文件名从synthetic19-20_*.csv 换成 redacted19-20_*.csv =====
files = {
    "N": "redacted19-20_N.csv",
    "U": "redacted19-20_U.csv",
    "Y": "redacted19-20_Y.csv",
}

dfs = {}
for name, filename in files.items():
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    dfs[name] = df

all_df = pd.concat(dfs.values(), ignore_index=True)
print(f"[Redacted] 合并后总记录数: {len(all_df)}")

# ===== 数据清洗：去掉U标签，只保留Y/N =====
clean_df = all_df[all_df["Predict"].isin(["Y", "N"])].reset_index(drop=True)
print(f"\n[Redacted] 清洗后记录数（去掉U）: {len(clean_df)}")
print(f"标签分布：\n{clean_df['Predict'].value_counts()}")
print(f"不平衡比例: {clean_df['Predict'].value_counts()['N'] / clean_df['Predict'].value_counts()['Y']:.1f} : 1")

# ===== 70/30 分层划分，保持类别比例一致，random_state保持42方便和synthetic版本对齐比较 =====
train_df, test_df = train_test_split(
    clean_df,
    test_size=0.3,
    stratify=clean_df["Predict"],
    random_state=42
)

print(f"\n[Redacted] 训练集: {len(train_df)} 条，标签分布：\n{train_df['Predict'].value_counts()}")
print(f"\n[Redacted] 测试集: {len(test_df)} 条，标签分布：\n{test_df['Predict'].value_counts()}")

# ===== 改动点2:所有输出文件名加 _redacted 后缀，不覆盖之前synthetic的结果 =====
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)
clean_df.to_csv(os.path.join(output_dir, "clean_full_redacted.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "train_redacted.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_redacted.csv"), index=False)

print(f"\n已保存到 {output_dir} 文件夹：clean_full_redacted.csv, train_redacted.csv, test_redacted.csv")

# ===== 新增：和synthetic版本做一次快速比对，方便你判断两个版本是否有实质差异 =====
try:
    synthetic_clean = pd.read_csv(os.path.join(output_dir, "clean_full.csv"))
    same_length = len(synthetic_clean) == len(clean_df)
    same_labels = (synthetic_clean["Predict"].values == clean_df["Predict"].values).all() if same_length else False
    same_text = (synthetic_clean["Chief Complaint"].astype(str).values ==
                 clean_df["Chief Complaint"].astype(str).values).all() if same_length else False
    print("\n===== Redacted vs Synthetic 快速比对 =====")
    print(f"记录数是否相同: {same_length} (synthetic={len(synthetic_clean)}, redacted={len(clean_df)})")
    if same_length:
        print(f"标签是否逐条相同: {same_labels}")
        print(f"Chief Complaint文本是否逐条相同: {same_text}")
        if not same_text:
            diff_count = (synthetic_clean["Chief Complaint"].astype(str).values !=
                          clean_df["Chief Complaint"].astype(str).values).sum()
            print(f"文本不同的记录数: {diff_count} / {len(clean_df)}")
except FileNotFoundError:
    print("\n（未找到之前的clean_full.csv，跳过与synthetic的比对）")