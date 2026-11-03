"""Build training/labeled_windows.csv from the DEAM dataset.

For every DEAM song it cuts 5 s windows (from second 15 on, where DEAM annotations start),
computes the same columns your analyzer writes (bpm, bpm_std, db_mean, db_std, pitch_hz,
voiced_pct, arousal, valence, mode_score ...) and labels each window from DEAM's human
arousal/valence ratings.  Then run:  python train_model.py

Usage (from the project folder, next to mood_core.py):
  python build_deam_training.py --audio "D:/DEAM/MEMD_audio" --ann "D:/DEAM/DEAM_Annotations"
  python build_deam_training.py --audio ... --ann ... --limit 50      # quick test on 50 songs
Options: --db-offset -15  (shift DEAM dBFS toward your LabVIEW dB scale, see README note)
         --workers 4      (parallel songs; pitch tracking is the slow part)
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mood_core import NOTE_NAMES, detect_key, load_config, resolve, rule_mood

SR = 22050
HOP = 512
WIN = 5            # seconds, must match window_sec in config.json
START = 15         # DEAM annotations start at 15 s
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ------------------------------------------------------------ label mapping
def deam_label(a, v, a0, v0, neutral_band, hi):
    """Map human arousal/valence (-1..1) to the analyzer's mood names.
    a0/v0 re-centre DEAM (its ratings sit slightly positive overall)."""
    da, dv = a - a0, v - v0
    if abs(da) < neutral_band and abs(dv) < neutral_band:
        return "neutral"
    if da >= 0:
        if dv >= -0.05:
            return "energetic" if da > hi else "upbeat"
        return "tense" if da > hi else "melancholic"
    return "calm" if dv >= -0.05 else "melancholic"


# ------------------------------------------------------------ audio features
def song_features(path, db_offset, cfg, max_windows):
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    n_win = min(int((len(y) / sr - START) // WIN), max_windows)
    if n_win <= 0:
        return []

    fps = sr / HOP
    # frame-level tracks computed once per song, then sliced per window
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6)) + db_offset
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                                 sr=sr, frame_length=2048, hop_length=HOP)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo_t = librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=HOP, aggregate=None)
    pc_all = np.where(voiced & ~np.isnan(f0), np.round(69 + 12 * np.log2(f0 / 440.0)) % 12, np.nan)

    rows = []
    for k in range(n_win):
        t0, t1 = START + k * WIN, START + (k + 1) * WIN
        i0, i1 = int(t0 * fps), int(t1 * fps)
        sl = slice(i0, min(i1, len(db)))
        d = db[sl]
        f = f0[sl]
        vo = voiced[sl]
        tp = tempo_t[sl]
        pitch_hz = float(np.nanmedian(f[vo])) if vo.any() else np.nan
        bpm = float(np.median(tp)) if len(tp) else np.nan
        bpm = bpm if cfg["bpm_valid_min"] <= bpm <= cfg["bpm_valid_max"] else np.nan

        # key/mode from the trailing key_context_sec seconds, like analyzer.py
        c0 = max(0, int((t1 - cfg["key_context_sec"]) * fps))
        seg = pc_all[c0:i1]
        seg = seg[~np.isnan(seg)].astype(int)
        res = detect_key(seg) if len(seg) >= cfg["min_key_samples"] else None
        key, mode, conf = (res[0], res[1], res[2]) if res else ("", "", np.nan)

        db_mean = float(np.mean(d))
        a, v, _, mode_score = rule_mood(bpm, db_mean, pitch_hz, mode, conf, cfg)
        rows.append({
            "window_idx": k, "t0": t0,
            "bpm": bpm, "bpm_std": float(np.std(tp)) if len(tp) else np.nan,
            "db_mean": db_mean, "db_std": float(np.std(d)),
            "pitch_hz": pitch_hz, "voiced_pct": float(vo.mean() * 100) if len(vo) else np.nan,
            "key": key, "mode": mode, "key_conf": conf,
            "arousal": a, "valence": v, "mode_score": mode_score,
        })
    return rows


def process(args):
    sid, path, db_offset, cfg, max_windows = args
    try:
        return sid, song_features(path, db_offset, cfg, max_windows), None
    except Exception as e:  # one bad file must not stop the run
        return sid, [], str(e)


# ---------------------------------------------------------------------- main
def find_annotation(ann_root, name):
    if not Path(ann_root).exists():
        sys.exit(f"The --ann folder does not exist: {ann_root}\nExtract DEAM_Annotations.zip and point --ann at the extracted folder.")
    hits = [p for p in Path(ann_root).rglob(name) if "per each rater" not in str(p)]
    if not hits:
        sys.exit(f"Could not find {name} under {ann_root}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="folder with DEAM mp3 files (named <song_id>.mp3)")
    ap.add_argument("--ann", required=True, help="folder you unzipped DEAM_Annotations into")
    ap.add_argument("--out", default="training/labeled_windows.csv")
    ap.add_argument("--limit", type=int, default=0, help="only process this many songs (testing)")
    ap.add_argument("--max-windows", type=int, default=12, help="cap per song so full-length songs don't dominate")
    ap.add_argument("--db-offset", type=float, default=0.0, help="added to DEAM dB to match your LabVIEW scale")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--a0", type=float, default=0.15)
    ap.add_argument("--v0", type=float, default=0.10)
    ap.add_argument("--neutral-band", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.20, help="arousal above centre that counts as 'energetic/tense'")
    a = ap.parse_args()
    cfg = load_config()
    if cfg["window_sec"] != WIN:
        print(f"WARNING: config window_sec={cfg['window_sec']} but this script uses {WIN}s windows")

    if not Path(a.audio).exists():
        sys.exit(f"The --audio folder does not exist: {a.audio}\nPoint --audio at the folder that contains 2.mp3, 3.mp3 ...")
    A = pd.read_csv(find_annotation(a.ann, "arousal.csv")).set_index("song_id")
    V = pd.read_csv(find_annotation(a.ann, "valence.csv")).set_index("song_id")
    files = {}
    for p in Path(a.audio).rglob("*"):
        if p.suffix.lower() in AUDIO_EXT and p.stem.isdigit():
            files[int(p.stem)] = p
    ids = [i for i in A.index if i in files and i in V.index]
    print(f"{len(files)} audio files found, {len(ids)} match the annotations")
    if a.limit:
        ids = ids[: a.limit]
    if not ids:
        sys.exit("No matching songs. Audio files must be named like 2.mp3, 3.mp3 ...")

    jobs = [(i, files[i], a.db_offset, cfg, a.max_windows) for i in ids]
    out_rows, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for sid, rows, err in ex.map(process, jobs):
            done += 1
            if err:
                print(f"  skip {sid}: {err}")
            av, vv = A.loc[sid].dropna().to_numpy(), V.loc[sid].dropna().to_numpy()
            n = min(len(av), len(vv)) // 10          # 10 annotations (0.5 s) per 5 s window
            for r in rows:
                k = r["window_idx"]
                if k >= n:
                    continue
                ha, hv = av[k * 10:(k + 1) * 10].mean(), vv[k * 10:(k + 1) * 10].mean()
                r.update(session_id=f"deam_{sid}", song_id=sid, human_arousal=round(float(ha), 3),
                         human_valence=round(float(hv), 3),
                         label=deam_label(ha, hv, a.a0, a.v0, a.neutral_band, a.hi))
                out_rows.append(r)
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} songs, {len(out_rows)} windows")

    df = pd.DataFrame(out_rows)
    for c in ("bpm", "bpm_std", "db_mean", "db_std", "pitch_hz", "voiced_pct", "key_conf", "arousal", "valence", "mode_score"):
        df[c] = df[c].round(3)
    path = resolve(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nWrote {len(df)} labelled windows to {path}")
    print(df["label"].value_counts().to_string())
    print(f"\ndB range in these windows: {df['db_mean'].min():.1f} to {df['db_mean'].max():.1f} "
          f"(your data uses silence_db={cfg['silence_db']}, db_center={cfg['db_center']})")
    print("Next: python train_model.py")


if __name__ == "__main__":
    main()