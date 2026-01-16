import pandas as pd
df = pd.read_csv("data.csv")
print("-" * 30)
print("你 CSV 里的真实列名是:", df.columns.tolist())
print("-" * 30)