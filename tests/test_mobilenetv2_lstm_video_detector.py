"""
Unit and integration tests for the MobileNetV2-LSTM video fall detector.
Validates:
1. Frame preprocessing: BGR to RGB, 224x224 resizing, float32 range [0, 255].
2. Sequence sampling: 3 FPS downsampling, 15-frame windowing with stride 5.
3. Short video handling: Last-frame padding for clips < 15 sampled frames.
4. Model loading: Custom objects deserialization and singleton caching.
5. End-to-end detection: Prediction probabilities, threshold filtering, and AnalysisResult schema.
6. Detector factory: get_detector factory toggling between real and mock models.
"""

import os
import tempfile
import numpy as np
import pytest
import cv2

from mock_detector import (
    preprocess_frame,
    extract_sampled_sequences,
    load_fall_model,
    MobileNetV2LSTMVideoDetector,
    MockFallDetector,
    get_detector,
    AnalysisResult,
    FallEvent
)


@pytest.fixture
def synthetic_video_path():
    """Generates a temporary 2.0-second 30 FPS synthetic video (60 frames)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
    # Create 60 frames (2.0s at 30 FPS)
    for i in range(60):
        # Vary color slightly across frames
        frame = np.full((240, 320, 3), fill_value=(i * 4) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    yield path

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


@pytest.fixture
def synthetic_short_video_path():
    """Generates a very short 0.2-second 30 FPS synthetic video (6 frames -> 1 sampled frame at 3 FPS)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
    for i in range(6):
        frame = np.full((240, 320, 3), fill_value=128, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    yield path

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# =============================================================================
# 1. PREPROCESSING TESTS
# =============================================================================

def test_preprocess_frame_shape_and_range():
    """Tests that preprocess_frame converts BGR to RGB, resizes to (224, 224), and preserves [0, 255] float32."""
    # Blue image in BGR: B=255, G=0, R=0
    bgr_img = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_img[:, :, 0] = 255

    processed = preprocess_frame(bgr_img)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    # In RGB, Red is channel 0 (should be 0) and Blue is channel 2 (should be 255.0)
    assert np.allclose(processed[:, :, 0], 0.0)
    assert np.allclose(processed[:, :, 2], 255.0)
    # Value range must be [0.0, 255.0] (NOT normalized to [0, 1] or [-1, 1], as model embeds mobilenetv2_preprocess)
    assert np.max(processed) <= 255.0
    assert np.min(processed) >= 0.0


# =============================================================================
# 2. SEQUENCE SAMPLING & WINDOWING TESTS
# =============================================================================

def test_extract_sampled_sequences_downsampling_and_padding(synthetic_video_path):
    """
    Video has 60 frames at 30 FPS.
    At 3 FPS downsampling, step = 30 / 3 = 10.
    Sampled frames = 60 / 10 = 6 frames.
    Since 6 < 15, padding repeats the last frame to yield exactly 1 sequence of 15 frames.
    """
    sequences, window_meta, v_info = extract_sampled_sequences(
        synthetic_video_path,
        sampling_fps=3.0,
        sequence_length=15,
        stride=5
    )

    assert v_info["fps"] == 30.0
    assert v_info["total_frames"] == 60
    assert len(sequences) == 1
    assert sequences.shape == (1, 15, 224, 224, 3)
    assert sequences.dtype == np.float32
    assert len(window_meta) == 1
    assert window_meta[0]["is_padded"] is True
    assert window_meta[0]["original_frames_count"] == 6


def test_extract_sampled_sequences_multi_window():
    """Generates a 6-second video (180 frames at 30 FPS -> 18 sampled frames at 3 FPS).
    With sequence_length=15 and stride=5:
    Windows:
    - Window 0: frames 0..14
    - Tail Window 1: frames 3..17 (final tail included)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (160, 120))
    for _ in range(180):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    try:
        sequences, window_meta, v_info = extract_sampled_sequences(
            path,
            sampling_fps=3.0,
            sequence_length=15,
            stride=5
        )
        assert len(sequences) >= 2
        assert sequences.shape[1:] == (15, 224, 224, 3)
        assert not window_meta[0]["is_padded"]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_extract_sampled_sequences_nonexistent_file():
    """Verifies FileNotFoundError on invalid video path."""
    with pytest.raises(FileNotFoundError):
        extract_sampled_sequences("nonexistent_video_file_xyz.mp4")


# =============================================================================
# 3. MODEL LOADING & CACHING TESTS
# =============================================================================

def test_load_fall_model_caching():
    """Verifies that load_fall_model loads the trained .keras weights and returns cached instance on subsequent calls."""
    model_path = "research_model/best_fall_model.keras"
    if not os.path.exists(model_path):
        pytest.skip(f"Model checkpoint not found at {model_path}")

    m1 = load_fall_model(model_path)
    assert m1 is not None
    assert m1.input_shape == (None, 15, 224, 224, 3)
    assert m1.output_shape == (None, 1)

    m2 = load_fall_model(model_path)
    # Must be identical cached instance in memory
    assert m1 is m2


def test_load_fall_model_missing_raises_error():
    """Verifies FileNotFoundError on non-existent model path."""
    with pytest.raises(FileNotFoundError):
        load_fall_model("nonexistent_path/fake_weights.keras")


# =============================================================================
# 4. END-TO-END DETECTOR TESTS
# =============================================================================

def test_mobilenetv2_lstm_video_detector_analyze_video(synthetic_video_path):
    """Verifies that MobileNetV2LSTMVideoDetector completes analysis, returns AnalysisResult, and populates telemetry."""
    model_path = "research_model/best_fall_model.keras"
    if not os.path.exists(model_path):
        pytest.skip(f"Model checkpoint not found at {model_path}")

    progress_events = []

    def progress_callback(pct, msg):
        progress_events.append((pct, msg))

    detector = MobileNetV2LSTMVideoDetector(
        model_path=model_path,
        threshold=0.50,
        sampling_fps=3.0,
        sequence_length=15,
        stride=5
    )

    result = detector.analyze_video(synthetic_video_path, progress_callback=progress_callback)

    assert isinstance(result, AnalysisResult)
    assert result.video_name == os.path.basename(synthetic_video_path)
    assert result.fps == 30.0
    assert result.total_frames == 60
    assert result.duration_sec == 2.0
    assert result.processing_time_sec > 0
    assert len(result.telemetry_data) > 0
    assert "fall_probability" in result.telemetry_data[0]
    assert 0.0 <= result.telemetry_data[0]["fall_probability"] <= 1.0

    # Verify progress callback was triggered
    assert len(progress_events) >= 4
    assert progress_events[-1][0] == 1.0


def test_detector_threshold_filtering(synthetic_video_path):
    """
    Verifies that setting a threshold of 1.01 (impossible) results in fall_detected=False,
    while setting a threshold of -0.01 (guaranteed) triggers fall_detected=True.
    """
    model_path = "research_model/best_fall_model.keras"
    if not os.path.exists(model_path):
        pytest.skip(f"Model checkpoint not found at {model_path}")

    # Threshold > 1.0 ensures no falls triggered
    detector_high = MobileNetV2LSTMVideoDetector(model_path=model_path, threshold=1.01)
    res_high = detector_high.analyze_video(synthetic_video_path)
    assert res_high.fall_detected is False
    assert len(res_high.fall_events) == 0
    assert res_high.risk_level == "NORMAL"

    # Threshold < 0.0 ensures all sequences trigger falls
    detector_low = MobileNetV2LSTMVideoDetector(model_path=model_path, threshold=-0.01)
    res_low = detector_low.analyze_video(synthetic_video_path)
    assert res_low.fall_detected is True
    assert len(res_low.fall_events) > 0
    assert res_low.risk_level == "CRITICAL"
    assert res_low.fall_events[0].severity in ["CRITICAL", "HIGH", "MEDIUM"]


def test_short_video_graceful_handling(synthetic_short_video_path):
    """Verifies that very short videos (e.g. 6 frames / 0.2s) are handled gracefully via padding."""
    model_path = "research_model/best_fall_model.keras"
    if not os.path.exists(model_path):
        pytest.skip(f"Model checkpoint not found at {model_path}")

    detector = MobileNetV2LSTMVideoDetector(model_path=model_path, threshold=0.50)
    res = detector.analyze_video(synthetic_short_video_path)

    assert isinstance(res, AnalysisResult)
    assert res.total_frames == 6
    assert len(res.telemetry_data) == 1


# =============================================================================
# 5. FACTORY FUNCTION TESTS
# =============================================================================

def test_get_detector_factory():
    """Verifies that get_detector returns MobileNetV2LSTMVideoDetector when use_real_model=True, and MockFallDetector when False."""
    real_detector = get_detector(use_real_model=True)
    assert isinstance(real_detector, MobileNetV2LSTMVideoDetector)

    mock_detector = get_detector(use_real_model=False)
    assert isinstance(mock_detector, MockFallDetector)
