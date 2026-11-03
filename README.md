# Live Music Mood Dashboard

The LabVIEW VI appends readings to `data.csv`. The Node server watches that file, runs the trained Python models on new 5-second windows, writes JSON snapshots under `output/`, and serves the live dashboard.

```
LabVIEW VI -> data.csv -> server.js -> analyzer.py -> trained models
                                             |              |
                                      output/*.json <- dashboard HTML
```

## Run

Install the Python dependencies once:

```
pip install -r requirements.txt
```

Start the app from this folder:

```
npm start
```

Open [http://localhost:3000](http://localhost:3000). Keep the Node server running while the VI appends to `data.csv`. The page refreshes its JSON data every two seconds; analysis runs when the CSV changes, no more often than `min_interval_sec`. The dashboard colors change with the latest predicted mood.

If Python is not available as `python`, change `python_cmd` in `config.json` to `py`. Set `data_file` there to the VI's CSV path if it is not the `data.csv` in this folder. CSV headers are matched without regard to case or punctuation; `Timestamp`, `BPM`, `dB_Level`, `Pitch_Hz`, and `Note` are supported, along with the VI alarm columns.

If port 3000 is already in use, start on another port in PowerShell:

```
$env:PORT=3108
npm start
```

Then open `http://localhost:3108`. The browser must use the Node server URL so the page can access its `/api/dashboard` JSON endpoint; opening the HTML directly as a file will not connect to the VI data.

## Models and JSON

`config.json` enables trained inference by default. When available, `models/av_model.joblib` predicts arousal and valence, and its trained arousal/valence result determines the mood label. `models/mood_model.joblib` is used as the fallback mood classifier if the arousal/valence model is unavailable. Both use the feature names saved in their model bundles. Silent windows are labeled `silence` rather than sent to either model.

Each analysis writes:

- `output/dashboard.json`: model status, latest prediction, and trend windows used by the page.
- `output/windows.json`: time-window predictions and measurements.
- `output/latest.json`: the latest window prediction.
- `output/status.json`: analysis status and source-file details.

The dashboard uses `/api/dashboard`; JSON is also available at `/api/status`, `/api/latest`, and `/api/windows?limit=100`. Analysis settings, including the 5-second window, silence threshold, and file watcher interval, are in `config.json`.

The dashboard shows the latest note sample and current alarm flags (clipping, tempo, and off-key). Alarm events are also written to `output/alarms.csv`; the note shown as current comes from the newest VI row, while the recent-window note is shown separately.

The trained mood classifier currently supports the classes included in its saved model (`calm`, `energetic`, `melancholic`, and `tense`). The dashboard displays those model predictions directly.

## Connect the LabVIEW VI

1. In `config.json`, set `data_file` to the same CSV path the VI writes. Keep `output_dir` and `model_file` pointed at the project's `output` and `models` folders.
2. Configure the VI to append rows with a timestamp and BPM, dB level, pitch, and note fields. Keep the CSV header; do not overwrite it with each reading.
3. Start the server with `npm start`, then open its local URL in a browser.
4. Start the VI. The server watches the CSV, predicts mood from each new window with the trained model, and refreshes the JSON dashboard data. The page shows the current mood, measurements, and trends; it does not capture audio.

The supplied `models/mood_model.joblib` and `models/av_model.joblib` are used automatically with `use_ml_model: true`. `output/dashboard.json` is regenerated from the VI CSV on server startup and when that CSV changes.
