import pandas as pd

data_dir = r"D:\硕士毕业课题\dataset\sorted"

redacted_y = pd.read_csv(data_dir + r"\redacted19-20_Y.csv")
synthetic_y = pd.read_csv(data_dir + r"\synthetic19-20_Y.csv")

print("=== redacted19-20_Y.csv 前5行 ===")
print(redacted_y.head().to_string())

print("\n=== synthetic19-20_Y.csv 前5行 ===")
print(synthetic_y.head().to_string())

print("\n=== 两个文件的Chief Complaint是否完全相同 ===")
print((redacted_y["Chief Complaint"].values == synthetic_y["Chief Complaint"].values).all())