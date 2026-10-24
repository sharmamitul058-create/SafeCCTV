"""
Training Pipeline for MobileNetV2-LSTM Fall Detection.
Supports configurable dataset paths (both video files and extracted RGB image sequences like URFD),
Google Colab GPU training, and guarantees video-level split integrity (zero data leakage).
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

from .config import ResearchModelConfig
from .model import build_mobilenetv2_lstm
from .dataset import VideoSequenceDataset, VideoLevelSplitter
from .utils import setup_colab_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2-LSTM Fall Detection Model")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory containing fall/ and adl/ video folders or image sequence folders. Overrides config default.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training and validation (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate (default: 1e-4)")
    parser.add_argument("--lstm-units", type=int, default=64,
                        help="Number of LSTM recurrent units (default: 64)")
    parser.add_argument("--sampling-fps", type=float, default=3.0,
                        help="Target sampling rate in FPS (default: 3.0)")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="Native recording FPS of source video/sequence (default: 30.0 for URFD)")
    parser.add_argument("--sequence-length", type=int, default=15,
                        help="Number of frames per temporal window sequence (default: 15)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Temporal sliding window stride in frames (default: 5)")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable FP16 mixed precision for Colab GPU (T4/V100/A100)")
    parser.add_argument("--output-model", type=str, default=None,
                        help="Path to save the final trained .keras model")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save training checkpoints")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of videos reserved for validation (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of videos reserved for evaluation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def verify_zero_data_leakage(train_records, val_records, test_records) -> bool:
    """
    Verifies that no video entity or subject is shared across train, val, and test splits.
    """
    def get_identities(records):
        ids = set()
        for r in records:
            sid = r.get("subject_id") or os.path.basename(r["path"].rstrip("/\\"))
            ids.add(sid)
        return ids

    train_ids = get_identities(train_records)
    val_ids = get_identities(val_records)
    test_ids = get_identities(test_records)

    leak_train_val = train_ids.intersection(val_ids)
    leak_train_test = train_ids.intersection(test_ids)
    leak_val_test = val_ids.intersection(test_ids)

    has_leakage = bool(leak_train_val or leak_train_test or leak_val_test)
    if has_leakage:
        print("[DATA LEAKAGE WARNING] Overlapping entities detected across splits!")
        if leak_train_val:
            print(f"  • Train <-> Val overlap: {leak_train_val}")
        if leak_train_test:
            print(f"  • Train <-> Test overlap: {leak_train_test}")
        if leak_val_test:
            print(f"  • Val <-> Test overlap: {leak_val_test}")
    else:
        print("[Verification] ZERO data leakage confirmed across all video/subject entities.")

    return not has_leakage


def train_model(config: ResearchModelConfig):
    """
    Executes the training workflow:
    1. Sets up GPU / Colab environment.
    2. Discovers videos / image sequence folders and splits at VIDEO level (no data leakage).
    3. Extracts 224x224 RGB frames at target FPS into 15-frame sequences.
    4. Trains MobileNetV2-LSTM with callbacks.
    5. Saves model artifacts.
    """
    print("=" * 70)
    print("  SafeCCTV Research: MobileNetV2-LSTM Fall Detection Training  ")
    print("=" * 70)

    # 1. Colab GPU Setup
    gpu_info = setup_colab_gpu(enable_mixed_precision=config.use_mixed_precision)

    # 2. Video / Sequence Discovery
    data_loader = VideoSequenceDataset(config)
    video_records = data_loader.discover_video_files(config.data_dir)

    print(f"\n[Dataset Discovery] Scanning directory: {config.data_dir}")
    print(f"  • Discovered {len(video_records)} total sequences/videos.")

    if not video_records:
        print(f"\n[Dataset Error] No valid video files or image sequence directories found in '{config.data_dir}'.")
        print("Expected folder layout for extracted RGB sequences (URFD):")
        print("  dataset_dir/")
        print("    ├── fall/")
        print("    │   ├── fall-01-cam0-rgb/  (contains *.png frames)")
        print("    │   └── fall-02-cam0-rgb/  (contains *.png frames)")
        print("    └── adl/")
        print("        ├── adl-01-cam0-rgb/   (contains *.png frames)")
        print("        └── adl-02-cam0-rgb/   (contains *.png frames)")
        print("Or video files (.mp4, .avi) inside 'fall/' and 'adl/' folders.")
        return None

    # Count breakdown
    n_fall = sum(1 for v in video_records if v["label"] == 1)
    n_adl = sum(1 for v in video_records if v["label"] == 0)
    n_img_seq = sum(1 for v in video_records if v.get("type") == "image_sequence")
    n_vid_files = sum(1 for v in video_records if v.get("type") == "video_file")

    print(f"  • Class distribution: {n_fall} Falls (positive), {n_adl} ADL (negative)")
    print(f"  • Source types:       {n_img_seq} Image Sequence Folders, {n_vid_files} Video Files")

    # 3. Video-Level Splitting (Zero Data Leakage)
    splitter = VideoLevelSplitter(
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed
    )
    train_videos, val_videos, test_videos = splitter.split_videos(video_records)

    print(f"\n[Video-Level Data Splitting]")
    print(f"  • Train sequences:      {len(train_videos)} ({sum(v['label'] for v in train_videos)} falls, {len(train_videos) - sum(v['label'] for v in train_videos)} ADL)")
    print(f"  • Validation sequences: {len(val_videos)} ({sum(v['label'] for v in val_videos)} falls, {len(val_videos) - sum(v['label'] for v in val_videos)} ADL)")
    print(f"  • Test sequences:       {len(test_videos)} ({sum(v['label'] for v in test_videos)} falls, {len(test_videos) - sum(v['label'] for v in test_videos)} ADL)")

    verify_zero_data_leakage(train_videos, val_videos, test_videos)

    # 4. Extract 15-frame sequences at target sampling FPS
    print(f"\n[Sequence Extraction] Extracting {config.sequence_length}-frame sequences at {config.sampling_fps} FPS (source: {config.source_fps} FPS)...")
    X_train, y_train = data_loader.load_dataset_split(train_videos, stride=config.stride)
    X_val, y_val = data_loader.load_dataset_split(val_videos, stride=config.stride)

    print(f"  • Train dataset: {len(X_train)} sequences, Shape: {X_train.shape}")
    print(f"  • Val dataset:   {len(X_val)} sequences, Shape: {X_val.shape}")

    if len(X_train) == 0:
        print("[Error] No training sequences could be extracted from the dataset.")
        return None

    # Compute class weights for imbalanced data
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    total_samples = len(y_train)

    if n_pos > 0 and n_neg > 0:
        weight_for_0 = (1.0 / n_neg) * (total_samples / 2.0)
        weight_for_1 = (1.0 / n_pos) * (total_samples / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"  • Balanced class weights: 0 (ADL): {weight_for_0:.2f}, 1 (Fall): {weight_for_1:.2f}")
    else:
        class_weight = {0: 1.0, 1: 1.0}

    # 5. Build Model
    print("\n[Model Construction] Initializing MobileNetV2-LSTM architecture...")
    model = build_mobilenetv2_lstm(config=config, compile_model=True)
    model.summary()

    # 6. Callbacks
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.output_model_path) or ".", exist_ok=True)

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_fall_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(X_val) > 0 else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss" if len(X_val) > 0 else "loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if len(X_val) > 0 else "loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Convert to tf.data.Dataset
    ds_train = data_loader.create_tf_dataset(X_train, y_train, batch_size=config.batch_size, is_training=True)
    ds_val = data_loader.create_tf_dataset(X_val, y_val, batch_size=config.batch_size, is_training=False) if len(X_val) > 0 else None

    # 7. Model Training
    print(f"\n[Training] Starting training for {config.epochs} epochs...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # 8. Save Model
    model.save(config.output_model_path)
    print(f"\n[Saved Model] Successfully saved model to: {config.output_model_path}")

    # Save test video list for evaluate.py
    if len(test_videos) > 0:
        test_meta_path = os.path.join(config.checkpoint_dir, "test_split_metadata.json")
        with open(test_meta_path, "w") as f:
            json.dump(test_videos, f, indent=2)
        print(f"[Test Split] Test video paths saved for evaluation: {test_meta_path}")

    return model


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = ResearchModelConfig()

    if cli_args.data_dir:
        cfg.data_dir = cli_args.data_dir
    if cli_args.epochs:
        cfg.epochs = cli_args.epochs
    if cli_args.batch_size:
        cfg.batch_size = cli_args.batch_size
    if cli_args.lr:
        cfg.learning_rate = cli_args.lr
    if cli_args.lstm_units:
        cfg.lstm_units = cli_args.lstm_units
    if cli_args.sampling_fps:
        cfg.sampling_fps = cli_args.sampling_fps
    if cli_args.source_fps:
        cfg.source_fps = cli_args.source_fps
    if cli_args.sequence_length:
        cfg.sequence_length = cli_args.sequence_length
    if cli_args.stride:
        cfg.stride = cli_args.stride
    if cli_args.mixed_precision:
        cfg.use_mixed_precision = True
    if cli_args.output_model:
        cfg.output_model_path = cli_args.output_model
    if cli_args.checkpoint_dir:
        cfg.checkpoint_dir = cli_args.checkpoint_dir
    if cli_args.val_split is not None:
        cfg.val_split = cli_args.val_split
    if cli_args.test_split is not None:
        cfg.test_split = cli_args.test_split
    if cli_args.seed:
        cfg.random_seed = cli_args.seed

    train_model(cfg)
