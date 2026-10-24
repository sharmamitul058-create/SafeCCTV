"""
Unit and integration tests for the research_model module.
Validates:
1. Configuration parameters (224x224 RGB, 15-frame sequences, 3 FPS).
2. MobileNetV2-LSTM model input/output tensor shapes and forward pass.
3. Strict video-level data splitting to guarantee zero data leakage.
4. Exactness of classification metrics (precision, recall, F1, confusion matrix).
5. Inference latency measurement.
6. Sequence extraction from video with 3 FPS downsampling and 15-frame windowing.
"""

import os
import tempfile
import numpy as np
import pytest
import cv2

from research_model.config import ResearchModelConfig
from research_model.model import build_mobilenetv2_lstm
from research_model.dataset import VideoLevelSplitter, VideoSequenceDataset
from research_model.utils import compute_classification_metrics, measure_inference_latency


def test_config_dimensions():
    """Validates required model specifications (224x224, 15 frames, 3 FPS)."""
    cfg = ResearchModelConfig()
    assert cfg.frame_height == 224
    assert cfg.frame_width == 224
    assert cfg.frame_channels == 3
    assert cfg.sequence_length == 15
    assert cfg.sampling_fps == 3.0
    assert cfg.input_shape == (15, 224, 224, 3)
    assert cfg.frame_shape == (224, 224, 3)


def test_mobilenetv2_lstm_architecture_and_forward_pass():
    """
    Constructs model with unweighted base (to avoid network download during test)
    and validates tensor shapes through a forward inference pass.
    """
    cfg = ResearchModelConfig(mobilenet_weights=None, lstm_units=16, dense_units=8)
    model = build_mobilenetv2_lstm(config=cfg, compile_model=True)

    # Input shape: (None, 15, 224, 224, 3)
    assert model.input_shape == (None, 15, 224, 224, 3)
    # Output shape: (None, 1) sigmoid fall probability
    assert model.output_shape == (None, 1)

    # Test forward pass with batch size of 2
    dummy_input = np.random.uniform(0.0, 255.0, size=(2, 15, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (2, 1)
    # Output must be valid probabilities in [0.0, 1.0]
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_video_level_splitter_prevents_data_leakage():
    """
    Verifies that video-level splitting partitions distinct videos across splits
    so that no video file ever appears in more than one partition.
    """
    splitter = VideoLevelSplitter(val_split=0.2, test_split=0.2, random_seed=42)

    # Create 30 synthetic video records (15 falls, 15 ADL)
    videos = []
    for i in range(15):
        videos.append({"path": f"/data/fall/video_{i:02d}.mp4", "label": 1, "subject_id": f"sub_{i:02d}"})
    for i in range(15):
        videos.append({"path": f"/data/adl/video_{i:02d}.mp4", "label": 0, "subject_id": f"sub_{i:02d}"})

    train_v, val_v, test_v = splitter.split_videos(videos)

    train_paths = set(v["path"] for v in train_v)
    val_paths = set(v["path"] for v in val_v)
    test_paths = set(v["path"] for v in test_v)

    # Zero overlap check between any pair of splits
    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected between Train and Val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected between Train and Test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected between Val and Test!"

    # Total videos must match input count
    assert len(train_paths) + len(val_paths) + len(test_paths) == len(videos)

    # Both classes must be represented in the train set
    train_labels = [v["label"] for v in train_v]
    assert 0 in train_labels and 1 in train_labels


def test_classification_metrics_exactness():
    """Validates metrics calculation (precision, recall, F1, confusion matrix)."""
    # y_true: 4 samples [1, 1, 0, 0]
    # y_pred_probs: [0.9 (TP), 0.3 (FN), 0.7 (FP), 0.2 (TN)] with threshold=0.5
    # y_pred_binary: [1, 0, 1, 0]
    # TP=1, FN=1, FP=1, TN=1
    # Precision = 1 / (1 + 1) = 0.5
    # Recall = 1 / (1 + 1) = 0.5
    # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    # Accuracy = (1 + 1) / 4 = 0.5

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.3, 0.7, 0.2])

    metrics = compute_classification_metrics(y_true, y_pred, threshold=0.5)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1


def test_measure_inference_latency_benchmark():
    """Validates latency profiler function."""
    cfg = ResearchModelConfig(mobilenet_weights=None, lstm_units=8, dense_units=4)
    model = build_mobilenetv2_lstm(config=cfg, compile_model=False)

    latencies = measure_inference_latency(
        model=model,
        sequence_shape=(1, 15, 224, 224, 3),
        warmup_runs=2,
        benchmark_runs=3
    )

    assert "avg_latency_ms" in latencies
    assert latencies["avg_latency_ms"] > 0
    assert latencies["throughput_fps"] > 0
    assert latencies["benchmark_runs"] == 3


def test_video_sequence_extractor_and_sampling():
    """
    Creates a temporary synthetic 30 FPS video with 60 frames (2.0 seconds).
    At 3 FPS downsampling:
    Expected sampled frames = 60 / 10 = 6 frames.
    Since 6 < 15 frames, padding by repeating the last frame must produce exactly
    a sequence of length 15 with shape (1, 15, 224, 224, 3).
    """
    cfg = ResearchModelConfig()
    loader = VideoSequenceDataset(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_video_path = os.path.join(tmp_dir, "test_synthetic.mp4")

        # Write 60 synthetic frames (640x480) at 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        for _ in range(60):
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(dummy_frame)
        writer.release()

        # Extract frames at 3 FPS
        frames = loader.extract_frames_at_fps(test_video_path)
        assert len(frames) == 6  # 60 frames / (30/3) = 6 frames
        assert frames.shape[1:] == (224, 224, 3)

        # Extract 15-frame sequences with padding
        X, y = loader.extract_sequences_from_video(test_video_path, label=1, stride=5)
        assert X.shape == (1, 15, 224, 224, 3)
        assert y.shape == (1, 1)
        assert y[0, 0] == 1.0


def test_natural_frame_sorting():
    """Validates that natural sorting preserves true chronological frame ordering."""
    from research_model.dataset import natural_sort_key

    # Standard lexicographical sort would fail on these (putting 10 before 2, 100 before 20)
    scrambled = [
        "frame_100.png",
        "frame_1.png",
        "frame_20.png",
        "frame_2.png",
        "frame_10.png",
        "frame_3.png"
    ]
    expected = [
        "frame_1.png",
        "frame_2.png",
        "frame_3.png",
        "frame_10.png",
        "frame_20.png",
        "frame_100.png"
    ]
    assert sorted(scrambled, key=natural_sort_key) == expected

    # Test URFD frame filenames
    urfd_frames = [
        "fall-01-cam0-rgb-010.png",
        "fall-01-cam0-rgb-002.png",
        "fall-01-cam0-rgb-001.png",
        "fall-01-cam0-rgb-020.png"
    ]
    expected_urfd = [
        "fall-01-cam0-rgb-001.png",
        "fall-01-cam0-rgb-002.png",
        "fall-01-cam0-rgb-010.png",
        "fall-01-cam0-rgb-020.png"
    ]
    assert sorted(urfd_frames, key=natural_sort_key) == expected_urfd


def test_urfd_multi_camera_event_level_zero_leakage():
    """
    Simulates URFD dual-camera capture: Camera 0 and Camera 1 record the same fall event.
    Verifies that VideoLevelSplitter binds both cameras to the same event partition,
    ensuring zero cross-camera and zero event leakage.
    """
    records = []
    # 10 Fall events, each with Camera 0 and Camera 1
    for i in range(1, 11):
        sid = f"fall-{i:02d}"
        records.append({"path": f"/data/fall/fall-{i:02d}-cam0-rgb", "label": 1, "subject_id": sid})
        records.append({"path": f"/data/fall/fall-{i:02d}-cam1-rgb", "label": 1, "subject_id": sid})

    # 10 ADL events (Camera 0)
    for i in range(1, 11):
        sid = f"adl-{i:02d}"
        records.append({"path": f"/data/adl/adl-{i:02d}-cam0-rgb", "label": 0, "subject_id": sid})

    splitter = VideoLevelSplitter(val_split=0.2, test_split=0.2, random_seed=42)
    train_set, val_set, test_set = splitter.split_videos(records)

    train_ids = set(r["subject_id"] for r in train_set)
    val_ids = set(r["subject_id"] for r in val_set)
    test_ids = set(r["subject_id"] for r in test_set)

    # 1. Zero event overlap across splits
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test!"

    # 2. For every fall event, verify Camera 0 and Camera 1 are in the EXACT SAME split
    for i in range(1, 11):
        cam0_path = f"/data/fall/fall-{i:02d}-cam0-rgb"
        cam1_path = f"/data/fall/fall-{i:02d}-cam1-rgb"

        in_train_0 = any(r["path"] == cam0_path for r in train_set)
        in_train_1 = any(r["path"] == cam1_path for r in train_set)
        in_val_0 = any(r["path"] == cam0_path for r in val_set)
        in_val_1 = any(r["path"] == cam1_path for r in val_set)
        in_test_0 = any(r["path"] == cam0_path for r in test_set)
        in_test_1 = any(r["path"] == cam1_path for r in test_set)

        assert in_train_0 == in_train_1, f"Fall-{i:02d} cam0 and cam1 separated across train split!"
        assert in_val_0 == in_val_1, f"Fall-{i:02d} cam0 and cam1 separated across val split!"
        assert in_test_0 == in_test_1, f"Fall-{i:02d} cam0 and cam1 separated across test split!"


def test_ancestor_path_label_integrity():
    """
    Verifies that directories containing 'fall' in ancestor folder names
    (e.g., 'E:/SafeCCTV/fall_project/dataset/adl/adl-01-cam0-rgb') do NOT corrupt
    ADL labels into false positives.
    """
    data_dir = "/home/user/deep_fall_detection_research/data"

    # ADL path under fall-named project directory
    adl_path = f"{data_dir}/adl/adl-01-cam0-rgb"
    assert VideoSequenceDataset.determine_label(adl_path, data_dir=data_dir) == 0

    # Fall path under same root
    fall_path = f"{data_dir}/fall/fall-01-cam0-rgb"
    assert VideoSequenceDataset.determine_label(fall_path, data_dir=data_dir) == 1

    # Flat sequence directory
    assert VideoSequenceDataset.determine_label(f"{data_dir}/adl-05-cam0-rgb", data_dir=data_dir) == 0
    assert VideoSequenceDataset.determine_label(f"{data_dir}/fall-05-cam0-rgb", data_dir=data_dir) == 1


def test_image_sequence_discovery_and_extraction():
    """
    Creates a temporary mock URFD directory containing PNG frames,
    verifies discovery, natural frame ordering, and 15-frame window extraction.
    """
    cfg = ResearchModelConfig(sampling_fps=3.0, source_fps=30.0, sequence_length=15, stride=5)
    loader = VideoSequenceDataset(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create fall and adl image sequence folders
        fall_seq_dir = os.path.join(tmp_dir, "fall", "fall-01-cam0-rgb")
        adl_seq_dir = os.path.join(tmp_dir, "adl", "adl-01-cam0-rgb")
        os.makedirs(fall_seq_dir)
        os.makedirs(adl_seq_dir)

        # Write 50 frames (approx 1.67s at 30 FPS)
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(1, 51):
            cv2.imwrite(os.path.join(fall_seq_dir, f"fall-01-cam0-rgb-{i:03d}.png"), dummy_img)
            cv2.imwrite(os.path.join(adl_seq_dir, f"adl-01-cam0-rgb-{i:03d}.png"), dummy_img)

        # 1. Discover sequences
        records = loader.discover_video_files(tmp_dir)
        assert len(records) == 2

        labels = {r["filename"]: r["label"] for r in records}
        assert labels["fall-01-cam0-rgb"] == 1
        assert labels["adl-01-cam0-rgb"] == 0

        # 2. Extract sequences from directory
        # 50 frames at 30 FPS sampled at 3 FPS (step = 10): 5 sampled frames
        # 5 sampled frames < 15 frames -> padded to exactly 15 frames
        X_fall, y_fall = loader.extract_sequences_from_video(fall_seq_dir, label=1)
        assert X_fall.shape == (1, 15, 224, 224, 3)
        assert y_fall.shape == (1, 1)
        assert y_fall[0, 0] == 1.0


def test_prepare_urfd_verification_utility():
    """Validates the verify_dataset function in prepare_urfd.py."""
    from research_model.prepare_urfd import verify_dataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        fall_seq = os.path.join(tmp_dir, "fall", "fall-01-cam0-rgb")
        adl_seq = os.path.join(tmp_dir, "adl", "adl-01-cam0-rgb")
        os.makedirs(fall_seq)
        os.makedirs(adl_seq)

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(fall_seq, "frame_001.png"), dummy_img)
        cv2.imwrite(os.path.join(fall_seq, "frame_002.png"), dummy_img)
        cv2.imwrite(os.path.join(adl_seq, "frame_001.png"), dummy_img)

        stats = verify_dataset(tmp_dir)
        assert stats["fall_sequences"] == 1
        assert stats["adl_sequences"] == 1
        assert stats["total_frames"] == 3
        assert stats["corrupted_images"] == 0

