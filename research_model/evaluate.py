"""
Evaluation and Benchmarking Pipeline for MobileNetV2-LSTM Fall Detection.
Evaluates on test video sequences (zero data leakage), computes precision,
recall, F1-score, confusion matrix, and measures real inference latency.
No metrics or results are fabricated.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from .config import ResearchModelConfig
from .dataset import VideoSequenceDataset, VideoLevelSplitter
from .utils import compute_classification_metrics, measure_inference_latency


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2-LSTM Fall Detection Model")
    parser.add_argument("--model-path", type=str, default="research_model/mobilenetv2_lstm_fall.keras",
                        help="Path to the trained .keras model file")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset root directory containing fall/ and adl/ subdirectories")
    parser.add_argument("--test-metadata", type=str, default=None,
                        help="Path to test_split_metadata.json generated during training")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--sampling-fps", type=float, default=3.0,
                        help="Target sampling rate in FPS (default: 3.0)")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="Native recording FPS of source video/sequence (default: 30.0 for URFD)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Temporal sliding window stride in frames (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification decision threshold (default: 0.5)")
    parser.add_argument("--output-json", type=str, default="research_model/evaluation_results.json",
                        help="File to save evaluation metrics as JSON")
    return parser.parse_args()


def evaluate_model(
    model_path: str,
    data_dir: Optional[str] = None,
    test_metadata_path: Optional[str] = None,
    threshold: float = 0.5,
    output_json_path: Optional[str] = None,
    config: Optional[ResearchModelConfig] = None
) -> dict:
    """
    Evaluates the model on test videos and benchmarks latency.
    """
    if config is None:
        config = ResearchModelConfig()

    print("=" * 70)
    print("  SafeCCTV Research: MobileNetV2-LSTM Evaluation & Benchmarking  ")
    print("=" * 70)

    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"[Evaluation Error] Model file not found at: {model_path}")
        print("Please train the model first with: python -m research_model.train")
        return {"error": f"Model not found at {model_path}"}

    print(f"\n[Model Loading] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Determine Test Videos
    data_loader = VideoSequenceDataset(config)
    test_records = []

    if test_metadata_path and os.path.exists(test_metadata_path):
        print(f"[Test Dataset] Loading test video split metadata from: {test_metadata_path}")
        with open(test_metadata_path, "r") as f:
            test_records = json.load(f)
    elif data_dir and os.path.exists(data_dir):
        # Look for explicit test directory: data_dir/test/fall and data_dir/test/adl
        explicit_test_dir = os.path.join(data_dir, "test")
        target_dir = explicit_test_dir if os.path.exists(explicit_test_dir) else data_dir

        all_videos = data_loader.discover_video_files(target_dir)
        if explicit_test_dir == target_dir:
            test_records = all_videos
        else:
            # Use same seed video splitter to isolate the test set
            splitter = VideoLevelSplitter(val_split=config.val_split, test_split=config.test_split, random_seed=config.random_seed)
            _, _, test_records = splitter.split_videos(all_videos)
    else:
        # Check default checkpoint location for test metadata
        default_meta = os.path.join(config.checkpoint_dir, "test_split_metadata.json")
        if os.path.exists(default_meta):
            with open(default_meta, "r") as f:
                test_records = json.load(f)

    if not test_records:
        print("[Dataset Error] No test video files identified for evaluation.")
        print("Provide --data-dir or --test-metadata with valid video paths.")
        return {"error": "No test videos available for evaluation."}

    print(f"[Test Dataset] Found {len(test_records)} test videos.")

    # 3. Extract Test Sequences (Zero Data Leakage)
    print(f"\n[Sequence Extraction] Extracting {config.sequence_length}-frame sequences at {config.sampling_fps} FPS from test videos...")
    X_test, y_test = data_loader.load_dataset_split(test_records, stride=config.stride)

    if len(X_test) == 0:
        print("[Dataset Error] Failed to extract any valid sequences from test videos.")
        return {"error": "Zero sequences extracted from test videos."}

    print(f"  • Extracted Test sequences: {len(X_test)} (Shape: {X_test.shape})")
    print(f"  • True Falls: {int(np.sum(y_test == 1))}, True ADL: {int(np.sum(y_test == 0))}")

    # 4. Predict
    print("\n[Inference] Running model prediction on test sequences...")
    y_pred_probs = model.predict(X_test, batch_size=config.batch_size, verbose=1)

    # 5. Compute Metrics (Precision, Recall, F1, Confusion Matrix)
    metrics = compute_classification_metrics(y_true=y_test, y_pred_probs=y_pred_probs, threshold=threshold)

    # 6. Measure Real Inference Latency
    print("\n[Latency Benchmarking] Measuring inference latency (25 runs, 5 warmup)...")
    latency_stats = measure_inference_latency(
        model=model,
        sequence_shape=(1, config.sequence_length, config.frame_height, config.frame_width, 3),
        warmup_runs=5,
        benchmark_runs=25
    )

    # Print Clean Summary Table
    print("\n" + "=" * 55)
    print("               EVALUATION REPORT               ")
    print("=" * 55)
    print(f"Total Test Sequences:     {metrics['samples']}")
    print(f"Decision Threshold:       {metrics['threshold']:.2f}")
    print(f"Accuracy:                 {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision:                {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall:                   {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-Score:                 {metrics['f1_score']:.4f}")
    print("-" * 55)
    cm = metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"                  Predicted ADL (0)   Predicted Fall (1)")
    print(f"  Actual ADL (0)  {cm['tn']:<18}  {cm['fp']:<18} (TN, FP)")
    print(f"  Actual Fall (1) {cm['fn']:<18}  {cm['tp']:<18} (FN, TP)")
    print("-" * 55)
    print("Inference Speed:")
    print(f"  Average Latency:        {latency_stats['avg_latency_ms']} ms / sequence")
    print(f"  p95 Latency:            {latency_stats['p95_latency_ms']} ms / sequence")
    print(f"  Throughput:             {latency_stats['throughput_fps']} sequences / sec (FPS)")
    print("=" * 55)

    full_results = {
        "model_path": model_path,
        "metrics": metrics,
        "latency": latency_stats
    }

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\n[Saved Metrics] Saved full evaluation report to: {output_json_path}")

    return full_results


if __name__ == "__main__":
    args = parse_args()
    cfg = ResearchModelConfig()
    if args.sampling_fps:
        cfg.sampling_fps = args.sampling_fps
    if args.source_fps:
        cfg.source_fps = args.source_fps
    if args.stride:
        cfg.stride = args.stride

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        test_metadata_path=args.test_metadata,
        threshold=args.threshold,
        output_json_path=args.output_json,
        config=cfg
    )
