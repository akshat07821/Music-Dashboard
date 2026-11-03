# 🎵 Music Dashboard - Product Requirements Document (PRD)

## Project Information

**Project Name:** Music Dashboard: Real-Time Audio Monitoring, Analytics & Alarm System

**Domain:** SCADA, Audio Signal Processing, Machine Learning

**Technologies Used:**
- LabVIEW
- Python
- CSV Telemetry Logging
- Power BI
- Web Dashboard

---

# 1. Project Overview

Music Dashboard is a SCADA-inspired real-time audio monitoring and analytics platform designed to bring industrial automation principles into the music and acoustic domain.

The system captures live audio signals, processes them in real time, detects pitch and tempo, monitors sound levels, logs telemetry data, generates alarms, and performs machine learning-based mood analysis through a decoupled LabVIEW-Python architecture.

---

# 2. Problem Statement

Traditional audio monitoring solutions typically focus on isolated tasks such as recording, spectrum analysis, or sound level monitoring. They lack:

- Integrated monitoring
- Alarm management
- Historical telemetry logging
- Machine learning analytics
- Unified visualization dashboards

This project addresses these gaps using a SCADA-based architecture for continuous audio supervision.

---

# 3. Objectives

### Primary Objectives

- Capture live audio continuously.
- Compute loudness (dB) in real time.
- Detect dominant pitch frequency.
- Convert frequencies into musical notes.
- Estimate tempo (BPM).
- Trigger alarms when thresholds are violated.
- Maintain telemetry logs.
- Generate ML-based mood analytics.
- Visualize data using dashboards.

---

# 4. Features

## Real-Time Audio Acquisition

- Continuous microphone input monitoring.
- Live waveform visualization.
- FFT-based frequency analysis.

## Signal Processing

- Decibel calculation.
- Pitch detection.
- Musical note identification.
- BPM estimation.

## Alarm Engine

Supports:

- Volume Clipping Alarm
- Tempo Out-of-Range Alarm
- Off-Key Detection Alarm

## Historical Logging

Stores:

- Timestamp
- BPM
- Loudness
- Frequency
- Note
- Alarm States

## Machine Learning Analytics

Python analytics engine generates:

- Mood Classification
- Arousal Score
- Valence Score
- Sliding Window Statistics

## Dashboard System

- LabVIEW HMI
- Web Dashboard
- Power BI Reporting

---

# 5. System Architecture

```
┌─────────────────┐
│ Live Audio Input│
└────────┬────────┘
         │
         ▼

┌─────────────────┐
│ LabVIEW DSP     │
│ Processing      │
└────────┬────────┘
         │
         ├──► HMI Dashboard
         │
         ├──► Alarm Engine
         │
         ▼

┌─────────────────┐
│ CSV Historian   │
│ Data Logging    │
└────────┬────────┘
         │
         ▼

┌─────────────────┐
│ Python Analytics│
│ Engine          │
└────────┬────────┘
         │
         ▼

┌─────────────────┐
│ Web Dashboard   │
│ Power BI        │
└─────────────────┘
```

---

# 6. Functional Requirements

## FR-01 Audio Acquisition

The system shall continuously capture audio from the microphone or audio interface.

### Acceptance Criteria

- Continuous acquisition.
- No frame drops.
- Stable execution.

---

## FR-02 Loudness Monitoring

The system shall calculate and display signal loudness.

### Acceptance Criteria

- Output in dB.
- Refresh in real time.
- Range: -60 dB to +10 dB.

---

## FR-03 Pitch Detection

The system shall identify dominant frequency.

### Acceptance Criteria

- Output in Hz.
- Accurate pitch tracking.
- Real-time updates.

---

## FR-04 Note Detection

The system shall convert pitch frequencies into musical notes.

### Supported Notes

```
C, C#, D, D#, E, F,
F#, G, G#, A, A#, B
```

---

## FR-05 Tempo Detection

The system shall estimate BPM.

### Acceptance Criteria

- BPM displayed continuously.
- Gauge indicator available.
- Range: 40–220 BPM.

---

## FR-06 Alarm Monitoring

### Volume Clipping Alarm

Triggered when:

```
dB_Level >= Threshold
```

### Tempo Alarm

Triggered when:

```
BPM < BPM_Min
OR
BPM > BPM_Max
```

### Off-Key Alarm

Triggered when:

```
Pitch deviation > tolerance
```

---

## FR-07 Telemetry Logging

The system shall save telemetry into CSV files.

### Logged Fields

- Timestamp
- BPM
- dB_Level
- Pitch_Hz
- Note
- Alarm_Clip
- Alarm_Tempo
- Alarm_Offkey

---

## FR-08 Mood Analytics

The analytics engine shall perform mood estimation.

### Outputs

- Arousal
- Valence
- Mood Label

Example Labels:

- Energetic
- Relaxed
- Excited
- Melancholic

---

# 7. Non-Functional Requirements

## Performance

| Requirement | Target |
|------------|---------|
| HMI Update Rate | ≥ 10 Hz |
| Logging Rate | 5 Hz |
| Alarm Response | < 100 ms |

## Reliability

- Continuous operation without crashes.
- No file corruption.

## Maintainability

- Modular architecture.
- Independent analytics layer.

## Scalability

- Future MQTT integration.
- OPC UA compatibility.
- Multi-room monitoring support.

---

# 8. Technology Stack

### Frontend

- LabVIEW HMI
- Web Dashboard

### Backend

- Python

### Data Storage

- CSV Files

### Analytics

- NumPy
- Pandas
- Signal Processing Algorithms

### Reporting

- Power BI
- JSON APIs

---

# 9. Success Metrics

| Metric | Goal |
|----------|----------|
| HMI Refresh Rate | ≥ 10 Hz |
| Logging Accuracy | 100% |
| Alarm Detection | <100 ms |
| Pitch Tracking | Stable |
| Data Loss | 0% |

---

# 10. Future Enhancements

- Chord Detection
- Polyphonic Pitch Analysis
- Cloud Database Integration
- MQTT Telemetry
- OPC UA Support
- Multi-Room Audio Monitoring
- Advanced Power BI Reports

---

# Author

- **Akshat Dubey**
- **VARNIKA JINDAL**
- **JAINAM PATEL**
- **DIVANSHI WADHWA**

MBA Tech (Data Science) | NMIMS

---

## Repository Description

A SCADA-inspired real-time audio monitoring system built using LabVIEW and Python that performs live DSP analysis, telemetry logging, alarm management, and machine learning-based mood analytics.
