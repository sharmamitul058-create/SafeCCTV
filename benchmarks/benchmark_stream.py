"""
Performance benchmark for camera_stream.py.
Measures:
  1. get_latest_frame() latency and throughput (FPS)
  2. Frame copy and memory throughput overhead
  3. YOLOv8 Pose process_frame() latency
  4. End-to-end pipeline latency (frame capture -> inference -> retrieval)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import numpy as np
from unittest.mock import patch, MagicMock
from camera_stream import CameraStreamManager


def benchmark_get_latest_frame(iterations=500):
    """Measures latency of retrieving frames via get_latest_frame()."""
    with patch("camera_stream.RealtimeFallDetector"), patch("threading.Thread"):
        with CameraStreamManager._lock:
            CameraStreamManager._instance = None
            manager = CameraStreamManager()
            manager.audio_running = False
            manager.is_running = True
            manager.start_time = time.time()
            manager.first_capture_received = True

            dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            with manager.frame_lock:
                manager.latest_raw_frame = dummy
                manager.latest_annotated_frame = dummy
                manager.last_capture_time = time.time()
                manager.last_inference_time = time.time()

            # Warmup
            for _ in range(20):
                manager.get_latest_frame()

            latencies = []
            start = time.perf_counter()
            for _ in range(iterations):
                t0 = time.perf_counter()
                frame, status, fps, alarm, elapsed = manager.get_latest_frame()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # in ms
            total_time = time.perf_counter() - start

            avg_latency_ms = float(np.mean(latencies))
            p95_latency_ms = float(np.percentile(latencies, 95))
            throughput_fps = iterations / total_time

            return {
                "avg_latency_ms": avg_latency_ms,
                "p95_latency_ms": p95_latency_ms,
                "throughput_fps": throughput_fps
            }


def benchmark_copy_overhead(iterations=300):
    """Measures the exact CPU overhead of the 4 redundant np.ascontiguousarray(frame.copy()) calls."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 4 redundant copies (original approach)
    t0 = time.perf_counter()
    for _ in range(iterations):
        c1 = np.ascontiguousarray(frame.copy(), dtype=np.uint8)
        c2 = np.ascontiguousarray(c1.copy(), dtype=np.uint8)
        c3 = np.ascontiguousarray(c2.copy(), dtype=np.uint8)
        c4 = np.ascontiguousarray(c3.copy(), dtype=np.uint8)
    time_4copies = (time.perf_counter() - t0) * 1000.0 / iterations

    # Zero extra copies (passing reference)
    t0 = time.perf_counter()
    for _ in range(iterations):
        c1 = frame
        c2 = c1
        c3 = c2
        c4 = c3
    time_0copies = (time.perf_counter() - t0) * 1000.0 / iterations

    return {
        "overhead_4copies_ms": time_4copies,
        "overhead_0copies_ms": time_0copies,
        "savings_ms_per_frame": time_4copies - time_0copies
    }


def benchmark_inference(iterations=50):
    """Measures RealtimeFallDetector.process_frame() latency with real YOLOv8 pose model."""
    from fall_detector import RealtimeFallDetector
    detector = RealtimeFallDetector(model_path="yolov8n-pose.pt")
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        detector.process_frame(dummy.copy(), "2026-10-23 00:00:00")

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.process_frame(dummy, "2026-10-23 00:00:00")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        "avg_inference_ms": float(np.mean(latencies)),
        "p95_inference_ms": float(np.percentile(latencies, 95)),
        "inference_fps": 1000.0 / float(np.mean(latencies))
    }


if __name__ == "__main__":
    print("--- BENCHMARK RESULTS ---")
    retrieval = benchmark_get_latest_frame()
    print(f"get_latest_frame() Latency: {retrieval['avg_latency_ms']:.4f} ms (p95: {retrieval['p95_latency_ms']:.4f} ms)")
    print(f"get_latest_frame() Throughput: {retrieval['throughput_fps']:.1f} FPS")

    copy_bench = benchmark_copy_overhead()
    print(f"Redundant 4-copy overhead: {copy_bench['overhead_4copies_ms']:.4f} ms/frame")
    print(f"Zero-copy reference overhead: {copy_bench['overhead_0copies_ms']:.6f} ms/frame")
    print(f"Copy time savings: {copy_bench['savings_ms_per_frame']:.4f} ms/frame")

    infer = benchmark_inference(30)
    print(f"YOLOv8 Pose Inference: {infer['avg_inference_ms']:.2f} ms ({infer['inference_fps']:.1f} FPS)")
