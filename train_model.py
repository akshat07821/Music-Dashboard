"""Train your own mood classifier on windows you have labelled.

1. Run the analyzer so output/windows.csv exists.
2. Copy output/windows.csv to training/labeled_windows.csv
3. Open it in Excel and fill the new 'label' column for as many rows as you can
   (e.g. calm, upbeat, energetic, tense, melancholic, neutral). Aim for 30+ per label,
   from at least 3 different sessions. Leave unknown rows blank.
4. python train_model.py
   -> models/mood_model.joblib  (the analyzer uses it automatically on the next run)
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from mood_core import FEATURES, load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="training/labeled_windows.csv")
    a = ap.parse_args()
    cfg = load_config()

    path = resolve(a.labels)
    if not path.exists():
        sys.exit(f"Not found: {path}\nCopy output/windows.csv there and fill the 'label' column first.")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        sys.exit("The file needs a 'label' column.")
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"nan": ""})
    df = df[(df["label"] != "") & (df["label"] != "silence")]

    counts = df["label"].value_counts()
    keep = counts[counts >= 5].index
    if len(keep) < len(counts):
        print("Dropping labels with fewer than 5 rows:", ", ".join(sorted(set(counts.index) - set(keep))))
    df = df[df["label"].isin(keep)]
    if df["label"].nunique() < 2:
        sys.exit("Need at least 2 labels with 5+ rows each.")

    X, y = df[FEATURES], df["label"]
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])

    # honest evaluation: keep whole sessions together when there are enough of them
    groups = df["session_id"] if "session_id" in df.columns else None
    if groups is not None and groups.nunique() >= 3:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        print(f"Cross-validation by session ({groups.nunique()} sessions):")
    else:
        k = int(min(5, y.value_counts().min()))
        pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=42))
        print("Cross-validation (random split; use 3+ sessions for a stricter test):")
    print(classification_report(y, pred, zero_division=0))

    model.fit(X, y)
    out = resolve(cfg["model_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "classes": list(model.classes_)}, out)
    print("Saved", out)
    imp = sorted(zip(FEATURES, model.named_steps["rf"].feature_importances_), key=lambda x: -x[1])
    print("Feature importance:", ", ".join(f"{k}={v:.2f}" for k, v in imp))


if __name__ == "__main__":
    main()
