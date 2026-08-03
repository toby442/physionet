import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_dir = r"D:\硕士毕业课题\dataset\sorted"

files = {
    "N": "synthetic19-20_N.csv",
    "U": "synthetic19-20_U.csv",
    "Y": "synthetic19-20_Y.csv",
}

dfs = {}
for name, filename in files.items():
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    dfs[name] = df

all_df = pd.concat(dfs.values(), ignore_index=True)
print(f"合并后总记录数: {len(all_df)}")

# ===== 数据清洗：去掉U标签，只保留Y/N =====
clean_df = all_df[all_df["Predict"].isin(["Y", "N"])].reset_index(drop=True)
print(f"\n清洗后记录数（去掉U）: {len(clean_df)}")
print(f"标签分布：\n{clean_df['Predict'].value_counts()}")
print(f"不平衡比例: {clean_df['Predict'].value_counts()['N'] / clean_df['Predict'].value_counts()['Y']:.1f} : 1")

# ===== 70/30 分层划分，保持类别比例一致 =====
train_df, test_df = train_test_split(
    clean_df,
    test_size=0.3,
    stratify=clean_df["Predict"],  # 分层抽样，保证训练集测试集里Y/N比例一致
    random_state=42  # 固定随机种子，保证结果可复现
)

print(f"\n训练集: {len(train_df)} 条，标签分布：\n{train_df['Predict'].value_counts()}")
print(f"\n测试集: {len(test_df)} 条，标签分布：\n{test_df['Predict'].value_counts()}")

# ===== 保存清洗后的数据，方便后续步骤直接读取 =====
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)
clean_df.to_csv(os.path.join(output_dir, "clean_full.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print(f"\n已保存到 {output_dir} 文件夹：clean_full.csv, train.csv, test.csv")