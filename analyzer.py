"""Reads the LabVIEW data.csv, computes sessions/windows/mood and writes Power BI-ready CSVs.

Safe to run repeatedly: sessions found in data.csv are recomputed and replaced,
sessions that are no longer in data.csv are kept (so history survives if LabVIEW
overwrites the file for a new run).
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from mood_core import apply_model, build_windows, load_config, load_raw, resolve

TS = "%Y-%m-%d %H:%M:%S"
WIN_COLS = ["session_id", "window_start", "window_end", "n_samples", "bpm", "bpm_std", "db_mean",
            "db_max", "db_std", "pitch_hz", "voiced_pct", "top_note", "key", "mode", "key_conf",
            "clip_events", "tempo_events", "arousal", "valence", "mode_score",
            "mood_rule", "arousal_ml", "valence_ml", "mood_ml", "mood"]
ALARM_COLS = ["session_id", "timestamp", "alarm_type", "bpm", "db_level", "pitch_hz"]


def atomic_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # file open in Power BI / Excel
            time.sleep(0.5)
    raise PermissionError(f"Could not replace {path} - close it in Excel/Power BI and retry")


def atomic_json(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_dashboard_json(out, status, windows, latest):
    records = json.loads(windows.to_json(orient="records")) if not windows.empty else []
    latest_record = json.loads(latest.to_json(orient="records"))[0] if not latest.empty else None
    dashboard = {
        "generated_at": status["generated_at"],
        "source_rows": status["source_rows"],
        "sessions_in_source": status["sessions_in_source"],
        "model_used": status["model_used"],
        "model": status.get("model"),
        "latest": latest_record,
        "windows": records[-1200:],
    }
    atomic_json(dashboard, out / "dashboard.json")
    atomic_json(dashboard["windows"], out / "windows.json")
    atomic_json(latest_record, out / "latest.json")


def mode_of(s):
    s = s.dropna().astype(str)
    s = s[s.str.len() > 0]
    return s.mode().iat[0] if len(s) else ""


def summarize(w, win_sec):
    rows = []
    for sid, d in w.groupby("session_id"):
        active = d[d["mood"] != "silence"]
        start, end = d["window_start"].min(), d["window_end"].max()
        dur = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60
        rows.append({
            "session_id": sid,
            "session_label": pd.to_datetime(start).strftime("%Y-%m-%d %H:%M"),
            "start": start, "end": end, "duration_min": round(dur, 2),
            "windows": len(d), "active_windows": len(active),
            "avg_bpm": round(d["bpm"].mean(), 1), "avg_db": round(d["db_mean"].mean(), 2),
            "max_db": round(d["db_max"].max(), 2),
            "clip_events": int(d["clip_events"].sum()), "tempo_events": int(d["tempo_events"].sum()),
            "avg_arousal": round(active["arousal"].mean(), 3), "avg_valence": round(active["valence"].mean(), 3),
            "top_mood": mode_of(active["mood"]), "top_note": mode_of(active["top_note"]),
            "main_key": mode_of(active["key"]),
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def merge_history(new, path, key_cols_str, new_ids):
    if path.exists():
        try:
            old = pd.read_csv(path, dtype={"session_id": str})
            old = old[~old["session_id"].isin(new_ids)]
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    return new


def main():
    cfg = load_config()
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    status = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "source_rows": 0,
              "sessions_in_source": 0, "warnings": [], "model_used": False, "latest": None}

    df, warns = load_raw(cfg)
    status["warnings"] += warns
    status["source_rows"] = int(len(df))

    if df.empty:
        atomic_json(status, out / "status.json")
        status["model"] = None
        write_dashboard_json(out, status, pd.DataFrame(), pd.DataFrame())
        print("No data yet:", "; ".join(warns) or "-")
        return

    w, alarms = build_windows(df, cfg)
    status["model_used"] = apply_model(w, cfg, status["warnings"])
    status["model"] = cfg["av_model_file"] if cfg["use_av_model"] else cfg["model_file"]
    new_ids = set(w["session_id"])
    status["sessions_in_source"] = len(new_ids)

    # format + round
    w["window_start"] = w["window_start"].dt.strftime(TS)
    w["window_end"] = w["window_end"].dt.strftime(TS)
    for c, n in (("bpm", 1), ("bpm_std", 2), ("db_mean", 2), ("db_max", 2), ("db_std", 2),
                 ("pitch_hz", 1), ("voiced_pct", 1), ("key_conf", 3), ("arousal", 3),
                 ("valence", 3), ("mode_score", 3), ("arousal_ml", 3), ("valence_ml", 3)):
        w[c] = w[c].round(n)
    w = w[WIN_COLS]
    alarms["timestamp"] = alarms["timestamp"].dt.strftime(TS)
    alarms = alarms[ALARM_COLS].round({"bpm": 1, "db_level": 2, "pitch_hz": 1})

    all_w = merge_history(w, out / "windows.csv", None, new_ids).sort_values(["window_start"]).reset_index(drop=True)
    all_a = merge_history(alarms, out / "alarms.csv", None, new_ids).sort_values("timestamp").reset_index(drop=True)
    sessions = summarize(all_w, cfg["window_sec"])

    atomic_csv(all_w, out / "windows.csv")
    atomic_csv(all_a, out / "alarms.csv")
    atomic_csv(sessions, out / "sessions.csv")

    last = w.iloc[-1]
    current = df.iloc[-1]
    current_note = str(current["note"]).strip()
    if current_note.lower() in ("", "none", "n/a", "nan", "-", "--"):
        current_note = ""
    active_alarms = [
        name for column, name in (("clip", "Clipping"), ("tempo", "Tempo"), ("offkey", "Off-key"))
        if int(current[column]) == 1
    ]
    latest_window = w.tail(1).copy()
    latest_window["current_note"] = current_note
    latest_window["active_alarms"] = [active_alarms]
    status["latest"] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                        for k, v in last[["session_id", "window_start", "mood", "mood_rule", "bpm",
                                          "db_mean", "pitch_hz", "top_note", "key"]].to_dict().items()}
    status["latest"]["current_note"] = current_note
    status["latest"]["active_alarms"] = active_alarms
    status["totals"] = {"sessions": int(len(sessions)), "windows": int(len(all_w)), "alarms": int(len(all_a))}
    status["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    dashboard_windows = all_w.tail(1200)
    write_dashboard_json(out, status, dashboard_windows, latest_window)
    atomic_json(status, out / "status.json")
    print(f"OK rows={len(df)} sessions_in_source={len(new_ids)} windows_total={len(all_w)} "
          f"latest_mood={status['latest']['mood']} ml={'yes' if status['model_used'] else 'no'}")
    for m in status["warnings"]:
        print("WARNING:", m)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
