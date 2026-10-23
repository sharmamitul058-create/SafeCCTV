"""
Focused unit tests for tracking webcam capture health separately from YOLO inference health in camera_stream.py.
Verifies that:
  1. Slow or lagging YOLO inference does NOT mark a working camera as FROZEN.
  2. Stale webcam capture (>3.5s without camera frames) correctly transitions connection_status to FROZEN.
  3. Connection status is CONNECTING until the first capture frame is received, then transitions to LIVE immediately.
  4. Fallback to raw frame works when inference has not yet produced an annotated frame.
  5. get_health_status() reports capture_healthy vs inference_healthy separately and thread-safely.
"""

import time
import threading
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from camera_stream import CameraStreamManager


@pytest.fixture
def stream_manager():
    """Create a CameraStreamManager with audio thread and detector mocked to avoid hardware deps."""
    with patch("camera_stream.RealtimeFallDetector") as MockDetector, \
         patch("threading.Thread"):
        # Reset singleton instance for clean test isolation
        with CameraStreamManager._lock:
            CameraStreamManager._instance = None
            manager = CameraStreamManager()
            manager.detector = MagicMock()
            manager.audio_running = False
    return manager


class TestCaptureVsInferenceHealth:

    def test_slow_inference_does_not_mark_working_camera_as_frozen(self, stream_manager):
        """Webcam is actively delivering frames (last_capture_time is fresh),
        but YOLO inference is very slow (last_inference_time is > 10s old).
        The camera stream must remain LIVE, NOT FROZEN."""
        now = time.time()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with stream_manager.frame_lock:
            stream_manager.is_running = True
            stream_manager.start_time = now - 20.0
            stream_manager.first_capture_received = True
            stream_manager.first_frame_received = True
            # Capture is active (updated 50ms ago)
            stream_manager.last_capture_time = now - 0.05
            stream_manager.latest_raw_frame = dummy_frame
            # Inference is lagging heavily (10s ago)
            stream_manager.last_inference_time = now - 10.0
            stream_manager.last_frame_time = stream_manager.last_inference_time
            stream_manager.latest_annotated_frame = dummy_frame

        frame, status, fps, alarm, elapsed = stream_manager.get_latest_frame()

        assert status == "LIVE", f"Camera should be LIVE when capture is active, but got {status}"
        health = stream_manager.get_health_status()
        assert health["capture_healthy"] is True, "Capture should be reported as healthy"
        assert health["inference_healthy"] is False, "Inference should be reported as unhealthy/lagging"
        assert health["inference_age_sec"] is not None and health["inference_age_sec"] >= 9.0

    def test_stale_capture_marks_camera_as_frozen(self, stream_manager):
        """When webcam capture stops receiving frames for > 3.5s,
        the stream must be marked FROZEN."""
        now = time.time()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with stream_manager.frame_lock:
            stream_manager.is_running = True
            stream_manager.start_time = now - 20.0
            stream_manager.first_capture_received = True
            # Camera stopped delivering frames 4.0 seconds ago
            stream_manager.last_capture_time = now - 4.0
            stream_manager.last_inference_time = now - 4.0
            stream_manager.latest_raw_frame = dummy_frame
            stream_manager.latest_annotated_frame = dummy_frame

        frame, status, fps, alarm, elapsed = stream_manager.get_latest_frame()

        assert status == "FROZEN", f"Expected FROZEN when capture stops, got {status}"
        health = stream_manager.get_health_status()
        assert health["capture_healthy"] is False

    def test_connecting_before_first_capture(self, stream_manager):
        """Before the first frame is read from hardware, connection_status must be CONNECTING."""
        with stream_manager.frame_lock:
            stream_manager.is_running = True
            stream_manager.first_capture_received = False
            stream_manager.connection_status = "CONNECTING"

        frame, status, fps, alarm, elapsed = stream_manager.get_latest_frame()
        assert status == "CONNECTING"

    def test_first_capture_activates_live_immediately(self, stream_manager):
        """As soon as first capture arrives, connection_status becomes LIVE even if
        first YOLO inference has not finished."""
        now = time.time()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with stream_manager.frame_lock:
            stream_manager.is_running = True
            stream_manager.start_time = now
            stream_manager.first_capture_received = True
            stream_manager.first_frame_received = False  # inference not done yet
            stream_manager.last_capture_time = now
            stream_manager.last_inference_time = 0.0
            stream_manager.latest_raw_frame = dummy_frame
            stream_manager.latest_annotated_frame = None

        frame, status, fps, alarm, elapsed = stream_manager.get_latest_frame()

        assert status == "LIVE"
        assert frame is not None, "Should fall back to raw frame when annotated is not ready"

    def test_fallback_to_raw_frame_when_annotated_frame_none(self, stream_manager):
        """get_latest_frame(return_raw=False) falls back to latest_raw_frame if annotated is None."""
        raw = np.full((100, 100, 3), 42, dtype=np.uint8)

        with stream_manager.frame_lock:
            stream_manager.is_running = True
            stream_manager.start_time = time.time()
            stream_manager.first_capture_received = True
            stream_manager.last_capture_time = time.time()
            stream_manager.latest_raw_frame = raw
            stream_manager.latest_annotated_frame = None

        frame, _, _, _, _ = stream_manager.get_latest_frame(return_raw=False)
        assert frame is not None
        assert np.array_equal(frame, raw)

    def test_thread_safe_concurrent_timestamp_updates(self, stream_manager):
        """Simulate concurrent capture and inference threads updating timestamps
        and reading status without race conditions or deadlocks."""
        stream_manager.is_running = True
        stream_manager.start_time = time.time()
        dummy = np.zeros((50, 50, 3), dtype=np.uint8)
        stop_event = threading.Event()
        errors = []

        def capture_worker():
            try:
                for _ in range(50):
                    if stop_event.is_set():
                        break
                    with stream_manager.frame_lock:
                        stream_manager.latest_raw_frame = dummy
                        stream_manager.raw_frame_id += 1
                        stream_manager.last_capture_time = time.time()
                        stream_manager.first_capture_received = True
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        def inference_worker():
            try:
                for _ in range(25):
                    if stop_event.is_set():
                        break
                    with stream_manager.frame_lock:
                        stream_manager.latest_annotated_frame = dummy
                        stream_manager.processed_frame_id += 1
                        stream_manager.last_inference_time = time.time()
                        stream_manager.last_frame_time = stream_manager.last_inference_time
                        stream_manager.first_frame_received = True
                    time.sleep(0.004)
            except Exception as e:
                errors.append(e)

        def reader_worker():
            try:
                for _ in range(50):
                    if stop_event.is_set():
                        break
                    stream_manager.get_latest_frame()
                    stream_manager.get_health_status()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=capture_worker),
            threading.Thread(target=inference_worker),
            threading.Thread(target=reader_worker)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
        stop_event.set()

        assert len(errors) == 0, f"Concurrent execution produced errors: {errors}"
        health = stream_manager.get_health_status()
        assert health["capture_healthy"] is True
        assert health["inference_healthy"] is True
        assert health["raw_frame_id"] > 0
        assert health["processed_frame_id"] > 0
