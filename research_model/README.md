# MobileNetV2–LSTM Fall Detection Research Module

A TensorFlow/Keras spatio-temporal deep learning research pipeline for continuous human fall detection from video streams and extracted RGB image sequence datasets (such as the University of Rzeszów Fall Detection Dataset — URFD).

---

## 🔬 Architectural Specification

- **Input Frame Shape:** `224 × 224 × 3` (RGB).
- **Temporal Sequence Length:** `15` frames per sequence (representing 5.0 seconds of real-time action at 3 FPS).
- **Sampling Rate:** `3 FPS` downsampled from native recording framerates (e.g. URFD 30 FPS $\rightarrow$ step = 10, sampling frames at index 0, 10, 20, 30...).
- **Window Padding:** Short video clips (< 15 sampled frames) are padded to 15 frames by repeating the final frame.
- **Feature Extractor:** Pretrained `MobileNetV2` (ImageNet weights) wrapped in a `TimeDistributed` layer yielding 1280-dimensional feature embeddings per frame: `(batch, 15, 1280)`.
- **Temporal Layer:** Recurrent `LSTM(units=64)` modeling posture transitions and sudden vertical acceleration dynamics across frames.
- **Classification Head:** Dense layer with ReLU activation, Dropout (0.4), and a Sigmoid output neuron predicting the probability of a fall ($0 = \text{ADL/Normal Activity}$, $1 = \text{Fall}$).
- **Video-Level Data Leakage Prevention:** Train/Val/Test splits are strictly partitioned by **video entity/subject** rather than frame sequences. Multi-camera viewpoints of the same event (e.g., `fall-01-cam0-rgb` and `fall-01-cam1-rgb`) are grouped under the same subject entity (`fall-01`) so they are never split across train, val, or test.

---

## 📁 Required Dataset Folder Structure

The dataset loader automatically discovers both **extracted RGB image sequence directories** (URFD format) and **standard video files** (`.mp4`, `.avi`, `.mov`).

### Recommended Structure for Extracted RGB Sequences (URFD)

```text
research_model/data/
├── fall/
│   ├── fall-01-cam0-rgb/
│   │   ├── fall-01-cam0-rgb-001.png
│   │   ├── fall-01-cam0-rgb-002.png
│   │   └── ...
│   ├── fall-01-cam1-rgb/
│   │   ├── fall-01-cam1-rgb-001.png
│   │   └── ...
│   └── fall-30-cam0-rgb/
│       └── ...
└── adl/
    ├── adl-01-cam0-rgb/
    │   ├── adl-01-cam0-rgb-001.png
    │   ├── adl-01-cam0-rgb-002.png
    │   └── ...
    └── adl-40-cam0-rgb/
        └── ...
```

> [!NOTE]
> **Natural Chronological Sorting**: Frame files are sorted using natural alphanumeric ordering (`frame_1.png`, `frame_2.png`, ... `frame_10.png`), preventing standard lexicographic sorting from corrupting the temporal sequence order.

---

## ⚡ Colab-Compatible Dataset Preparation Workflow

A dedicated preparation script (`research_model/prepare_urfd.py`) is provided to automate downloading, extracting, and verifying URFD RGB image sequences directly on Google Colab or local machines.

### Step 1: Prepare URFD on Google Colab

Run in a Colab notebook cell:

```bash
# 1. Download and extract URFD RGB sequences (30 falls, 40 ADLs)
python -m research_model.prepare_urfd \
    --output-dir "research_model/data" \
    --zip-dir "research_model/downloads" \
    --max-falls 30 \
    --max-adls 40
```

To verify an existing extracted dataset:
```bash
python -m research_model.prepare_urfd \
    --output-dir "research_model/data" \
    --verify-only
```

---

## 🚀 Exact Training Commands (Do Not Run Yet)

### Google Colab (GPU Acceleration with T4/V100/A100)

```bash
python -m research_model.train \
    --data-dir "research_model/data" \
    --source-fps 30.0 \
    --sampling-fps 3.0 \
    --sequence-length 15 \
    --stride 5 \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0001 \
    --mixed-precision \
    --val-split 0.15 \
    --test-split 0.15 \
    --output-model "research_model/mobilenetv2_lstm_fall.keras" \
    --checkpoint-dir "research_model/checkpoints"
```

### Local Workstation (CPU / Local GPU)

```bash
python -m research_model.train \
    --data-dir "research_model/data" \
    --source-fps 30.0 \
    --sampling-fps 3.0 \
    --sequence-length 15 \
    --stride 5 \
    --epochs 20 \
    --batch-size 8 \
    --val-split 0.15 \
    --test-split 0.15
```

---

## 📊 Evaluation & Benchmarking Commands

```bash
python -m research_model.evaluate \
    --model-path "research_model/mobilenetv2_lstm_fall.keras" \
    --test-metadata "research_model/checkpoints/test_split_metadata.json" \
    --source-fps 30.0 \
    --sampling-fps 3.0 \
    --stride 5 \
    --threshold 0.5 \
    --output-json "research_model/evaluation_results.json"
```

### Metrics Reported:
1. **Precision:** $\frac{TP}{TP + FP}$
2. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
3. **F1-Score:** Harmonic mean of precision and recall.
4. **Confusion Matrix:** Full 2×2 counts for True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).
5. **Inference Latency:** Average latency per 15-frame sequence (ms), 95th-percentile latency, and throughput (FPS).
