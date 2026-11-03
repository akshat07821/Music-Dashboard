"""Creates a fake data.csv that looks like the LabVIEW VI output, for testing without the VI.

  python make_sample_data.py                 -> data.csv (comma separated)
  python make_sample_data.py --tab           -> tab separated (LabVIEW's default for spreadsheet files)
  python make_sample_data.py --out other.csv
"""
import argparse
import math
import random
from datetime import datetime, timedelta

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR, MINOR = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]


def note_hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def session(start, minutes, hz_rate, tonic, scale, bpm_base, db_base, octave):
    rows, n = [], int(minutes * 60 * hz_rate)
    deg = 0
    for i in range(n):
        t = start + timedelta(seconds=i / hz_rate)
        if random.random() < 0.4:
            deg = max(0, min(len(scale) - 1, deg + random.choice([-2, -1, 1, 2])))
        midi = 12 * (octave + 1) + tonic + scale[deg]
        bpm = max(0, random.gauss(bpm_base, 3)) if random.random() > 0.08 else 0
        db = db_base + random.gauss(0, 3)
        if i % 400 == 399:
            db = -1.5                                  # clipping burst
        pitch = note_hz(midi) * (1 + random.gauss(0, 0.002))
        clip = 1 if db > -3 else 0
        tempo = 1 if bpm and (bpm < 80 or bpm > 160) else 0
        name = f"{NAMES[midi % 12]}{midi // 12 - 1}"
        rows.append((t, round(bpm, 1), round(db, 2), round(pitch, 2), name, clip, tempo))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data.csv")
    ap.add_argument("--tab", action="store_true")
    a = ap.parse_args()
    random.seed(7)
    d = "\t" if a.tab else ","
    now = datetime.now().replace(microsecond=0)
    rows = []
    # calm / sad: slow, quiet, A minor, low octave
    rows += session(now - timedelta(hours=3), 4, 4, 9, MINOR, 68, -42, 3)
    # energetic / happy: fast, loud, C major, high octave
    rows += session(now - timedelta(hours=1), 4, 4, 0, MAJOR, 138, -22, 5)
    # medium neutral-ish
    rows += session(now - timedelta(minutes=10), 4, 4, 7, MAJOR, 102, -34, 4)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(d.join(["Timestamp", "BPM", "dB_Level", "Pitch_Hz", "Note", "Alarm_Clipping", "Alarm_Tempo"]) + "\n")
        for t, bpm, db, pitch, note, clip, tempo in rows:
            f.write(d.join([t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], str(bpm), str(db), str(pitch),
                            note, str(clip), str(tempo)]) + "\n")
    print(f"Wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
