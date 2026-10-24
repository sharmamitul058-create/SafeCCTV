"""
Configuration parameters for MobileNetV2-LSTM Fall Detection Research Model.
Supports environment variable overrides and programmatic reconfiguration for Google Colab and local systems.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ResearchModelConfig:
    # Frame & Sequence Dimensions
    frame_height: int = 224
    frame_width: int = 224
    frame_channels: int = 3
    sequence_length: int = 15  # 15 frames per sequence
    sampling_fps: float = 3.0   # Sample frames at 3 FPS
    source_fps: float = 30.0    # Native source video/sequence FPS (URFD RGB is 30 FPS)
    stride: int = 5             # Sliding window step between 15-frame sequences

    # Model Hyperparameters
    mobilenet_weights: str = "imagenet"  # 'imagenet' or None
    trainable_base_layers: int = 0       # 0 = frozen base; >0 unfreezes top N layers
    lstm_units: int = 64
    dropout_rate: float = 0.4
    dense_units: int = 32
    learning_rate: float = 1e-4

    # Dataset & Training Hyperparameters
    batch_size: int = 8
    epochs: int = 25
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # Storage Paths
    data_dir: str = field(default_factory=lambda: os.environ.get("FALL_DATASET_DIR", "research_model/data"))
    checkpoint_dir: str = field(default_factory=lambda: os.environ.get("CHECKPOINT_DIR", "research_model/checkpoints"))
    log_dir: str = field(default_factory=lambda: os.environ.get("LOG_DIR", "research_model/logs"))
    output_model_path: str = field(default_factory=lambda: os.environ.get("OUTPUT_MODEL_PATH", "research_model/mobilenetv2_lstm_fall.keras"))

    # Colab & Acceleration
    use_mixed_precision: bool = False  # Set True for Tensor Core GPUs (T4, V100, A100)

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Returns (sequence_length, height, width, channels)."""
        return (self.sequence_length, self.frame_height, self.frame_width, self.frame_channels)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Returns (height, width, channels)."""
        return (self.frame_height, self.frame_width, self.frame_channels)
