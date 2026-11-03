"""Train two regression models (arousal, valence) on the DEAM windows and check
whether they beat your rule-based arousal/valence.

  python train_av_model.py

Reads training/labeled_windows_6class.csv (falls back to training/labeled_windows.csv).
Uses human_arousal / human_valence from DEAM as the targets, and the same feature
columns as the analyzer. Cross-validation keeps whole songs together (no leakage).
Saves models/av_model.joblib.
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from mood_core import FEATURES, load_config, resolve

cfg = load_config()
path = resolve("training/labeled_windows_6class.csv")
if not path.exists():
    path = resolve("training/labeled_windows.csv")
if not path.exists():
    sys.exit("Training file not found - run build_deam_training.py first.")
df = pd.read_csv(path)
for c in ("human_arousal", "human_valence", "session_id"):
    if c not in df.columns:
        sys.exit(f"Column '{c}' missing - this file was not made by build_deam_training.py")
df = df[df["db_mean"] >= cfg["silence_db"]].dropna(subset=["human_arousal", "human_valence"])
print(f"{len(df)} windows from {df['session_id'].nunique()} songs ({path.name})\n")

X, groups = df[FEATURES], df["session_id"]
cv = GroupKFold(n_splits=5)
bundle = {"features": FEATURES}
results = {}

for name, target, rule in (("arousal", "human_arousal", "arousal"), ("valence", "human_valence", "valence")):
    y = df[target]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    ok = df[rule].notna()
    r_rule = np.corrcoef(df.loc[ok, rule], y[ok])[0, 1]
    r_ml = np.corrcoef(pred, y)[0, 1]
    results[name] = (r_rule, r_ml, r2_score(y, pred), mean_absolute_error(y, pred), pred)
    print(f"{name.upper():8s} correlation with human rating:  rules r={r_rule:.2f}   trained model r={r_ml:.2f}"
          f"   (R2={r2_score(y, pred):.2f}, mean error={mean_absolute_error(y, pred):.2f})")
    model.fit(X, y)
    bundle[name] = model
    bundle[name + "_centre"] = float(np.median(pred))   # used to split high/low when labelling

# how often the model gets high/low right (the practical question)
for name, target in (("arousal", "human_arousal"), ("valence", "human_valence")):
    y = df[target]
    centre = y.median()
    true_hi = y > centre
    pred_hi = results[name][4] > centre
    print(f"{name:8s} high-vs-low accuracy: {(true_hi == pred_hi).mean():.0%}  (50% = coin flip)")

out = resolve("models/av_model.joblib")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, out)
print(f"\nSaved {out}")
print("\nHow to read this: r near 0 = no skill, 0.3 = weak, 0.5 = decent, 0.7+ = good.")
