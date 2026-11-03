import pandas as pd, shutil
shutil.copy("training/labeled_windows.csv", "training/labeled_windows_6class.csv")
df = pd.read_csv("training/labeled_windows_6class.csv")
df = df[df["label"] != "neutral"]
df["label"] = df["label"].replace({"upbeat": "energetic"})
df.to_csv("training/labeled_windows.csv", index=False)
print(df["label"].value_counts())