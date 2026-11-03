"""Shared logic: config, tolerant CSV loading, session/window features,
key/mode detection, rule-based mood (valence-arousal) and optional ML model."""
import io
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

DEFAULTS = {
    "data_file": "data.csv", "output_dir": "output", "model_file": "models/mood_model.joblib",
    "window_sec": 5, "session_gap_sec": 30, "silence_db": -50,
    "bpm_valid_min": 30, "bpm_valid_max": 250,
    "bpm_center": 100, "bpm_span": 60, "db_center": -35, "db_span": 20,
    "key_context_sec": 20, "min_key_samples": 10, "use_ml_model": True,
    "use_av_model": True, "av_model_file": "models/av_model.joblib", "av_neutral_band": 0.03,
}

# Columns of windows.csv that the ML model is trained on
FEATURES = ["bpm", "bpm_std", "db_mean", "db_std", "pitch_hz", "voiced_pct",
            "arousal", "valence", "mode_score"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]  # Krumhansl-Kessler
MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

ALIASES = {
    "t": ["timestamp", "time", "datetime", "date"],
    "bpm": ["bpm", "tempo"],
    "db": ["dblevel", "db", "loudnessdb", "loudness"],
    "pitch": ["pitchhz", "pitch", "frequency", "freq"],
    "note": ["note", "notename"],
    "clip": ["alarmclipping", "clipping", "alarmclip"],
    "tempo_alarm": ["alarmtempo", "tempoalarm"],
    "offkey": ["alarmoffkey", "offkeyalarm"],
}


def load_config():
    cfg = dict(DEFAULTS)
    p = BASE / "config.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in raw.items() if not k.startswith("_")})
    return cfg


def resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else BASE / p


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ loading
def read_snapshot(path, tries=6):
    last = None
    for _ in range(tries):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8-sig", errors="replace")
        except OSError as e:  # file locked by LabVIEW while writing, or missing
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"Cannot read {path}: {last}")


def _flag(s):
    v = s.fillna("").str.strip().str.lower()
    return (v.isin(["true", "t", "yes", "y", "on"]) | (pd.to_numeric(v, errors="coerce").fillna(0) > 0)).astype(int)


def _parse_time(s):
    best = None
    for dayfirst in (False, True):
        t = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
        score = (int(t.notna().sum()), int((t.diff().dropna() >= pd.Timedelta(0)).sum()))
        if best is None or score > best[0]:
            best = (score, t)
    t = best[1]
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


def load_raw(cfg):
    """Return (DataFrame[t,bpm,db,pitch,note,clip,tempo], warnings)."""
    warnings = []
    path = resolve(cfg["data_file"])
    empty = pd.DataFrame(columns=["t", "bpm", "db", "pitch", "note", "clip", "tempo", "offkey"])
    if not path.exists():
        return empty, [f"Data file not found: {path}"]
    text = read_snapshot(path)
    if not text.endswith("\n"):  # last line may be half-written by LabVIEW
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    if not text.strip():
        return empty, ["Data file is empty"]

    first = text.splitlines()[0]
    delim = max([",", "\t", ";"], key=lambda d: first.count(d))
    if first.count(delim) == 0:
        delim = ","
    raw = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False,
                      skipinitialspace=True, on_bad_lines="skip")
    if raw.empty:
        return empty, ["Data file has a header but no rows yet"]

    heads = {re.sub(r"[\s_\-()]+", "", str(c).lower()): c for c in raw.columns}

    def col(key):
        for a in ALIASES[key]:
            if a in heads:
                return raw[heads[a]]
        return None

    def num(key):
        s = col(key)
        if s is None:
            warnings.append(f"Column for '{key}' not found - treated as missing")
            return pd.Series(np.nan, index=raw.index)
        if delim != ",":
            s = s.str.replace(",", ".", regex=False)  # decimal comma
        return pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame(index=raw.index)
    df["bpm"], df["db"], df["pitch"] = num("bpm"), num("db"), num("pitch")
    n = col("note")
    df["note"] = n.str.strip() if n is not None else ""
    c, tm, ok = col("clip"), col("tempo_alarm"), col("offkey")
    df["clip"] = _flag(c) if c is not None else 0
    df["tempo"] = _flag(tm) if tm is not None else 0
    df["offkey"] = _flag(ok) if ok is not None else 0

    ts = col("t")
    t = _parse_time(ts) if ts is not None else None
    if t is None or t.notna().mean() < 0.5:
        end = datetime.fromtimestamp(path.stat().st_mtime)
        t = pd.Series([end - timedelta(seconds=len(df) - 1 - i) for i in range(len(df))], index=df.index)
        warnings.append("Timestamps missing/unreadable - synthesised at 1 row per second ending at file modified time")
    df["t"] = t
    df = df.dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
    return df[["t", "bpm", "db", "pitch", "note", "clip", "tempo", "offkey"]], warnings


# ---------------------------------------------------------- pitch / key
_NOTE_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b\u266f\u266d]?)\s*(-?\d+)?\s*$")
_BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_pc(n):
    m = _NOTE_RE.match(n or "")
    if not m:
        return np.nan
    pc = _BASE_PC[m.group(1).upper()]
    if m.group(2) in ("#", "\u266f"):
        pc += 1
    elif m.group(2) in ("b", "\u266d"):
        pc -= 1
    return pc % 12


def _profiles():
    rows, meta = [], []
    for mode, prof in (("major", MAJ), ("minor", MIN)):
        for root in range(12):
            p = np.roll(np.array(prof, float), root)
            rows.append((p - p.mean()) / p.std())
            meta.append((root, mode))
    return np.array(rows), meta


_P, _META = _profiles()


def detect_key(pcs):
    """Krumhansl-Schmuckler key finding. pcs: array of pitch classes (ints)."""
    if len(pcs) == 0:
        return None
    hist = np.bincount(pcs, minlength=12).astype(float)
    if (hist > 0).sum() < 3 or hist.std() == 0:
        return None
    z = (hist - hist.mean()) / hist.std()
    r = _P @ z / 12.0
    i = int(np.argmax(r))
    root, mode = _META[i]
    return f"{NOTE_NAMES[root]} {mode}", mode, float(r[i])


# ------------------------------------------------------------- mood (rules)
def rule_mood(bpm, db, pitch_hz, mode, key_conf, cfg):
    """Russell valence-arousal estimate -> label. Mirrors the dashboard's moodFrom()."""
    bright = np.nan
    if pitch_hz and not np.isnan(pitch_hz) and pitch_hz > 0:
        midi = 69 + 12 * math.log2(pitch_hz / 440.0)
        bright = clamp((midi - 60) / 12.0, -1, 1)
    parts = []
    if not np.isnan(bpm):
        parts.append((0.5, clamp((bpm - cfg["bpm_center"]) / cfg["bpm_span"], -1, 1)))
    if not np.isnan(db):
        parts.append((0.3, clamp((db - cfg["db_center"]) / cfg["db_span"], -1, 1)))
    if not np.isnan(bright):
        parts.append((0.2, bright))
    if not parts:
        return np.nan, np.nan, "unknown", 0.0
    a = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    mc = clamp(((key_conf if not np.isnan(key_conf) else 0) - 0.4) / 0.5, 0, 1)
    mode_v = 1 if mode == "major" else -1 if mode == "minor" else 0
    v = 0.7 * mode_v * mc + 0.3 * (0 if np.isnan(bright) else bright)
    if abs(a) < 0.12 and abs(v) < 0.15:
        label = "neutral"
    elif a >= 0.12:
        label = ("energetic" if a > 0.4 else "upbeat") if v >= -0.1 else ("tense" if a > 0.3 else "melancholic")
    else:
        label = "calm" if v >= -0.1 else "melancholic"
    return a, v, label, mode_v * mc


# ------------------------------------------------------------------ windows
def _top_note(s):
    s = s.fillna("")
    s = s[(s.str.len() > 0) & ~s.str.lower().isin(["none", "n/a", "nan", "-", "--"])]
    return s.mode().iat[0] if len(s) else ""


def assign_sessions(df, cfg):
    gap = df["t"].diff().dt.total_seconds().gt(cfg["session_gap_sec"])
    grp = gap.cumsum()
    df["session_start"] = df.groupby(grp)["t"].transform("min")
    df["session_id"] = "S" + df["session_start"].dt.strftime("%Y%m%d_%H%M%S")
    return df


def build_windows(df, cfg):
    """Return (windows DataFrame, alarm events DataFrame) for the rows in df."""
    W = cfg["window_sec"]
    df = assign_sessions(df.copy(), cfg)
    df["offs"] = (df["t"] - df["session_start"]).dt.total_seconds()
    df["widx"] = (df["offs"] // W).astype(int)
    df["bpm_v"] = df["bpm"].where(df["bpm"].between(cfg["bpm_valid_min"], cfg["bpm_valid_max"]))
    df["pitch_v"] = df["pitch"].where(df["pitch"].between(40, 4200))

    pc = df["note"].map(note_to_pc)
    hz = df["pitch_v"]
    hz_pc = (np.round(69 + 12 * np.log2(hz.where(hz > 0) / 440.0)) % 12)
    df["pc"] = pc.where(pc.notna(), hz_pc)

    for flag, edge in (("clip", "clip_edge"), ("tempo", "tempo_edge"), ("offkey", "offkey_edge")):
        prev = df.groupby("session_id")[flag].shift(1).fillna(0)
        df[edge] = ((df[flag] == 1) & (prev == 0)).astype(int)

    g = df.groupby(["session_id", "widx"], sort=True)
    w = g.agg(
        session_start=("session_start", "first"),
        n_samples=("t", "size"),
        bpm=("bpm_v", "median"), bpm_std=("bpm_v", "std"),
        db_mean=("db", "mean"), db_max=("db", "max"), db_std=("db", "std"),
        pitch_hz=("pitch_v", "median"),
        voiced_pct=("pitch_v", lambda s: s.notna().mean() * 100),
        top_note=("note", _top_note),
        clip_events=("clip_edge", "sum"), tempo_events=("tempo_edge", "sum"),
    ).reset_index()
    w["window_start"] = w["session_start"] + pd.to_timedelta(w["widx"] * W, unit="s")
    w["window_end"] = w["window_start"] + pd.to_timedelta(W, unit="s")

    # key / mode from a trailing context of pitch classes
    sess = {sid: (d["offs"].to_numpy(), d["pc"].to_numpy()) for sid, d in df.groupby("session_id")}
    keys, modes, confs = [], [], []
    for sid, widx in zip(w["session_id"], w["widx"]):
        offs, pcs = sess[sid]
        end = (widx + 1) * W
        lo = np.searchsorted(offs, end - cfg["key_context_sec"], "left")
        hi = np.searchsorted(offs, end, "left")
        seg = pcs[lo:hi]
        seg = seg[~np.isnan(seg)].astype(int)
        res = detect_key(seg) if len(seg) >= cfg["min_key_samples"] else None
        keys.append(res[0] if res else "")
        modes.append(res[1] if res else "")
        confs.append(res[2] if res else np.nan)
    w["key"], w["mode"], w["key_conf"] = keys, modes, confs

    ar, va, lab, ms = [], [], [], []
    for r in w.itertuples():
        if not np.isnan(r.db_mean) and r.db_mean < cfg["silence_db"]:
            ar.append(np.nan); va.append(np.nan); lab.append("silence"); ms.append(0.0)
            continue
        a, v, l, m = rule_mood(r.bpm, r.db_mean, r.pitch_hz, r.mode, r.key_conf, cfg)
        ar.append(a); va.append(v); lab.append(l); ms.append(m)
    w["arousal"], w["valence"], w["mood_rule"], w["mode_score"] = ar, va, lab, ms

    alarms = []
    for flag, name in (("clip_edge", "Clipping"), ("tempo_edge", "Tempo"), ("offkey_edge", "Off-key")):
        e = df[df[flag] == 1]
        alarms.append(pd.DataFrame({
            "session_id": e["session_id"], "timestamp": e["t"], "alarm_type": name,
            "bpm": e["bpm"], "db_level": e["db"], "pitch_hz": e["pitch"]}))
    alarms = pd.concat(alarms).sort_values("timestamp").reset_index(drop=True)
    return w, alarms


def av_label(a, v, a_c, v_c, band):
    """Quadrant mood from predicted arousal/valence, relative to the training centres."""
    da, dv = a - a_c, v - v_c
    if abs(da) < band and abs(dv) < band:
        return "neutral"
    if da >= 0:
        return "energetic" if dv >= 0 else "tense"
    return "calm" if dv >= 0 else "melancholic"


def apply_model(w, cfg, warnings):
    """Final 'mood' = trained arousal/valence model if available (arousal_ml, valence_ml),
    else the optional classifier (mood_ml), else the rules (mood_rule)."""
    w["mood_ml"] = ""
    w["arousal_ml"] = np.nan
    w["valence_ml"] = np.nan
    mask = (w["mood_rule"] != "silence") & (w["mood_rule"] != "unknown")
    used = False

    pav = resolve(cfg["av_model_file"])
    if cfg["use_av_model"] and pav.exists():
        try:
            import joblib
            b = joblib.load(pav)
            if mask.any():
                X = w.loc[mask, b["features"]]
                a = b["arousal"].predict(X)
                v = b["valence"].predict(X)
                w.loc[mask, "arousal_ml"] = a
                w.loc[mask, "valence_ml"] = v
                w.loc[mask, "mood_ml"] = [av_label(x, y, b["arousal_centre"], b["valence_centre"],
                                                   cfg["av_neutral_band"]) for x, y in zip(a, v)]
            used = True
        except Exception as e:
            warnings.append(f"AV model could not be used ({e}); falling back")

    p = resolve(cfg["model_file"])
    if not used and cfg["use_ml_model"] and p.exists():
        try:
            import joblib
            bundle = joblib.load(p)
            if mask.any():
                w.loc[mask, "mood_ml"] = bundle["model"].predict(w.loc[mask, bundle["features"]])
            used = True
        except Exception as e:  # never let a bad model stop the pipeline
            warnings.append(f"ML model could not be used ({e}); using rule-based mood")
    w["mood"] = np.where(w["mood_ml"] != "", w["mood_ml"], w["mood_rule"])
    return used
