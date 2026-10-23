# SafeCCTV 🛡️

**Real-Time CCTV & Webcam Human Fall Monitoring & Instant Alarm Dashboard**

SafeCCTV is an intelligent video surveillance system designed to detect human falls in real time from live camera feeds (laptop webcams, network RTSP CCTV cameras, or prerecorded footage) using YOLOv8-pose estimation and motion/posture heuristics.

---

## 🌟 Key Features

- **Real-Time Pose & Fall Detection:** Powered by Ultralytics YOLOv8-Pose with custom multi-parameter heuristic validation (descent velocity, torso tilt angle, bounding box aspect ratio, and post-fall resting windows).
- **Multi-Source Support:** Works seamlessly with built-in laptop webcams (`Device 0`), external USB webcams, network RTSP IP cameras, and offline video uploads.
- **Instant Dual Alarms:**
  - **Local Hardware Siren:** Windows-native high-pitch siren (`winsound`).
  - **Browser Siren:** Automated HTML5 web audio siren playback with mute/unmute capabilities.
- **Incident Audit Logging:** Automatically logs verified incidents to `incidents/incident_log.csv` and captures photographic evidence snapshots.
- **Decoupled Architecture:** Thread-safe camera capture loop, background YOLO inference worker, and low-latency Streamlit UI fragment rendering.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sharmamitul058-create/SafetyCCTY.git
cd SafetyCCTY
pip install -r requirements.txt
```

### 2. Launching the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests & Benchmarks

Run the test suite:
```bash
python -m pytest
```

Run streaming performance benchmarks:
```bash
python benchmarks/benchmark_stream.py
```

---

## 📁 Repository Structure

```
├── app.py                      # Streamlit interactive monitoring dashboard
├── camera_stream.py            # Thread-safe camera worker & incident manager
├── fall_detector.py            # YOLOv8-Pose fall detection heuristics
├── mock_detector.py            # Video analysis & evaluation engine
├── benchmarks/                 # Pipeline latency & throughput benchmarks
├── tests/                      # Automated unit and integration test suite
├── incidents/                  # Incident snapshot evidence & CSV audit log
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```
