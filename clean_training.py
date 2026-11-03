import pandas as pd
from mood_core import load_config

cfg = load_config()
df = pd.read_csv("training/labeled_windows.csv")
before = len(df)
df = df[df["db_mean"] >= cfg["silence_db"]]
df.to_csv("training/labeled_windows.csv", index=False)
print(f"Kept {len(df)} of {before} windows")
print(df["label"].value_counts().to_string())