"""
Utilities for evaluation metrics calculation, inference latency profiling,
and Google Colab GPU configuration.
"""

import time
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)


def setup_colab_gpu(enable_mixed_precision: bool = False) -> Dict[str, Any]:
    """
    Configures TensorFlow for Google Colab GPU training.
    Enables memory growth to avoid allocating all VRAM upfront,
    and optionally activates FP16 mixed precision for modern Tensor Core GPUs.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_info = {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "device_names": [g.name for g in gpus],
        "mixed_precision": False
    }

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Colab GPU Setup] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(f"[Colab GPU Setup Warning] Error configuring memory growth: {e}")

        if enable_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                gpu_info["mixed_precision"] = True
                print("[Colab GPU Setup] Mixed Precision (mixed_float16) enabled for faster training.")
            except Exception as e:
                print(f"[Colab GPU Setup Warning] Mixed precision setup failed: {e}")
    else:
        print("[Colab GPU Setup] No GPU detected. Running on CPU.")

    return gpu_info


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics without fabrication:
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix (TN, FP, FN, TP)
    - Accuracy

    Args:
        y_true: Ground truth binary labels (0 = Normal/ADL, 1 = Fall).
        y_pred_probs: Model predicted probabilities in [0.0, 1.0].
        threshold: Classification decision boundary (default: 0.5).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_probs = np.asarray(y_pred_probs).ravel()

    if len(y_true) == 0:
        return {
            "error": "Empty dataset provided. Cannot compute metrics.",
            "samples": 0
        }

    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    # Scikit-learn metrics with zero_division handling
    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Text report
    report_str = classification_report(
        y_true,
        y_pred_binary,
        labels=[0, 1],
        target_names=["ADL/Normal (0)", "Fall (1)"],
        zero_division=0
    )

    return {
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]]
        },
        "classification_report": report_str
    }


def measure_inference_latency(
    model: tf.keras.Model,
    input_sample: Optional[np.ndarray] = None,
    sequence_shape: tuple = (1, 15, 224, 224, 3),
    warmup_runs: int = 5,
    benchmark_runs: int = 25
) -> Dict[str, float]:
    """
    Measures real inference latency (latency per sequence and throughput FPS).
    Performs GPU/CPU warmup runs before benchmark timing to eliminate cold-start overhead.
    """
    if input_sample is None:
        input_sample = np.random.uniform(0.0, 255.0, size=sequence_shape).astype(np.float32)

    # 1. Warmup passes
    for _ in range(warmup_runs):
        _ = model(input_sample, training=False)

    # 2. Benchmark timing
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model(input_sample, training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "avg_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(fps, 2)
    }
