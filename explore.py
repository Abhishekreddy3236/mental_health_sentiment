import pandas as pd

df = pd.read_csv("data/reddit_mental_health.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nNull values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)
