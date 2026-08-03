import pandas as pd

# 读取之前自动保存的完整对比结果
df = pd.read_csv("./processed_data/cv_all_8_methods_comparison.csv", index_col=0)

# 显示完整精确数字，不省略小数位
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df)

# 顺便导出一份更适合复制粘贴的简化表格(只保留mean，方便直接用)
simple_table = pd.DataFrame({
    "Recall": df["Recall_mean"].round(3).astype(str) + " ± " + df["Recall_std"].round(3).astype(str),
    "Precision": df["Precision_mean"].round(3).astype(str) + " ± " + df["Precision_std"].round(3).astype(str),
    "F1": df["F1_mean"].round(3).astype(str) + " ± " + df["F1_std"].round(3).astype(str),
    "ROC-AUC": df["AUC_mean"].round(3).astype(str) + " ± " + df["AUC_std"].round(3).astype(str),
})

print("\n\n===== methods compare sheet=====")
print(simple_table.to_string())

# 也保存成一个干净的CSV，方便你直接用Excel打开
simple_table.to_csv("./processed_data/final_summary_table.csv")
print("\n已保存简化表格到 processed_data/final_summary_table.csv")