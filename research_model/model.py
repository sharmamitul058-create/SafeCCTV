"""
MobileNetV2 + LSTM Spatio-Temporal Deep Learning Architecture for Fall Detection.
Uses MobileNetV2 (ImageNet pretrained) as a per-frame feature extractor wrapped in TimeDistributed,
followed by an LSTM layer to model temporal motion dynamics across 15-frame sequences.
"""

from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from .config import ResearchModelConfig


def build_mobilenetv2_lstm(
    config: Optional[ResearchModelConfig] = None,
    compile_model: bool = True
) -> tf.keras.Model:
    """
    Constructs the end-to-end MobileNetV2-LSTM model.

    Input:
        Sequence of frames with shape: (batch_size, sequence_length, height, width, channels)
        Default: (None, 15, 224, 224, 3)

    Architecture:
        1. Input: (None, 15, 224, 224, 3)
        2. Normalization: MobileNetV2 preprocess_input (-1.0 to 1.0)
        3. Feature Extraction: TimeDistributed(MobileNetV2(include_top=False, pooling='avg'))
           -> Output: (None, 15, 1280)
        4. Temporal Modeling: LSTM(units=lstm_units)
           -> Output: (None, lstm_units)
        5. Classification Head: Dense -> Dropout -> Dense(1, activation='sigmoid')
           -> Output: (None, 1) probability of fall (0 = Normal/ADL, 1 = Fall)
    """
    if config is None:
        config = ResearchModelConfig()

    seq_len = config.sequence_length
    h, w, c = config.frame_shape

    # 1. Sequence Input Layer
    sequence_input = layers.Input(
        shape=(seq_len, h, w, c),
        dtype=tf.float32,
        name="video_sequence_input"
    )

    # 2. Per-frame preprocessing layer
    # Scaled to [-1, 1] range expected by MobileNetV2
    preprocess_layer = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess"
    )

    # 3. Base CNN Feature Extractor
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=(h, w, c),
        include_top=False,
        weights=config.mobilenet_weights,
        pooling="avg"
    )
    base_cnn.trainable = False

    # Optional fine-tuning of top layers
    if config.trainable_base_layers > 0:
        base_cnn.trainable = True
        for layer in base_cnn.layers[:-config.trainable_base_layers]:
            layer.trainable = False

    # Wrap base CNN into a sequential or functional single-frame module
    frame_input = layers.Input(shape=(h, w, c), dtype=tf.float32, name="single_frame_input")
    x_frame = preprocess_layer(frame_input)
    feat_frame = base_cnn(x_frame)
    frame_feature_extractor = models.Model(
        inputs=frame_input,
        outputs=feat_frame,
        name="frame_feature_extractor"
    )

    # Apply feature extractor across all time steps
    temporal_features = layers.TimeDistributed(
        frame_feature_extractor,
        name="timedistributed_features"
    )(sequence_input)  # Shape: (None, 15, 1280)

    # 4. Temporal LSTM Layer (recurrent_dropout=0 enables cuDNN GPU speedup)
    lstm_out = layers.LSTM(
        units=config.lstm_units,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.0,
        name="lstm_temporal"
    )(temporal_features)  # Shape: (None, lstm_units)

    # 5. Fully Connected Classifier Head
    dense_1 = layers.Dense(
        config.dense_units,
        activation="relu",
        name="dense_fc"
    )(lstm_out)

    dropout_1 = layers.Dropout(
        config.dropout_rate,
        name="dropout_fc"
    )(dense_1)

    # Ensure float32 output for numerical stability with mixed precision
    fall_prob = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="fall_probability"
    )(dropout_1)

    model = models.Model(
        inputs=sequence_input,
        outputs=fall_prob,
        name="MobileNetV2_LSTM_FallDetector"
    )

    if compile_model:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

    return model
