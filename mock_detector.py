"""
SafeCCTV - Modular Detector Interface & Mock Detection Service
Designed for clean separation between Streamlit UI and Computer Vision detector engines.
"""

import os
import time
import cv2
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple


@dataclass
class FallEvent:
    timestamp: str           # e.g., "00:03.45"
    time_sec: float          # 3.45
    frame_idx: int           # frame number
    track_id: int            # person ID
    event_type: str          # "Rapid Descent", "Fall Confirmed", "Prolonged Prone"
    severity: str            # "HIGH", "MEDIUM", "LOW"
    confidence: float        # 0.0 - 1.0
    torso_angle: float       # degrees (0=upright, 90=horizontal)
    vertical_velocity: float # px/frame
    aspect_ratio: float      # W/H ratio
    condition_triggers: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    video_name: str
    video_path: str
    duration_sec: float
    fps: float
    total_frames: int
    resolution: str
    persons_tracked: int
    risk_level: str          # "CRITICAL", "WARNING", "NORMAL"
    fall_detected: bool
    fall_events: List[FallEvent]
    telemetry_data: List[Dict[str, Any]]
    annotated_video_path: Optional[str] = None
    processing_time_sec: float = 0.0
    model_name: str = "Mock Detector (Heuristic Simulation)"


class BaseFallDetector:
    """Abstract interface that both Mock and Real YOLO Detectors implement."""
    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> AnalysisResult:
        raise NotImplementedError("Subclasses must implement analyze_video")


class MockFallDetector(BaseFallDetector):
    """
    Simulates fall detection processing with realistic timing, progress reporting,
    telemetry logging, and annotated video generation.
    """
    def __init__(self, simulate_fall: bool = True):
        self.simulate_fall = simulate_fall

    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> AnalysisResult:
        start_time = time.time()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at: {video_path}")

        # Probe video properties with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if total_frames > 0 else 5.0
        cap.release()

        # Simulated pipeline stages for the progress bar
        stages = [
            (0.15, "Decoding video frames and validating streams..."),
            (0.35, "Running YOLO pose estimation (COCO 17 keypoints)..."),
            (0.60, "Tracking persons with persistent ByteTrack IDs..."),
            (0.80, "Evaluating posture angle and descent velocity heuristics..."),
            (0.95, "Synthesizing annotated overlay and compiling telemetry log..."),
            (1.00, "Analysis complete.")
        ]

        for prog, message in stages:
            if progress_callback:
                progress_callback(prog, message)
            time.sleep(0.35)  # Responsive, lightweight simulated latency

        # Generate realistic mock fall events
        fall_events = []
        telemetry = []

        # Determine if we simulate a fall event based on video length
        has_fall = self.simulate_fall and duration_sec >= 2.0
        fall_start_sec = min(3.2, max(1.0, duration_sec * 0.45))
        fall_start_frame = int(fall_start_sec * fps)

        # Generate synthetic frame-by-frame telemetry
        sample_step = max(1, total_frames // 40)
        for f_idx in range(0, total_frames, sample_step):
            t_sec = f_idx / fps
            timestamp_str = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"

            if has_fall and abs(t_sec - fall_start_sec) < 1.5:
                # Sudden fall motion window
                is_alert = True
                state = "FALL_ALERT"
                velocity = 44.8
                angle = 76.4
                aspect_ratio = 1.48
                hip_y = height * 0.78
            elif has_fall and (fall_start_sec - 0.5 <= t_sec < fall_start_sec):
                # Rapid descent initiation
                is_alert = False
                state = "RAPID_DESCENT"
                velocity = 38.2
                angle = 48.0
                aspect_ratio = 0.95
                hip_y = height * 0.58
            else:
                # Normal upright walking
                is_alert = False
                state = "NORMAL"
                velocity = 4.1
                angle = 12.3
                aspect_ratio = 0.42
                hip_y = height * 0.48

            telemetry.append({
                "frame": f_idx,
                "timestamp": timestamp_str,
                "time_sec": round(t_sec, 2),
                "track_id": 1,
                "hip_y": round(hip_y, 1),
                "hip_v_velocity": round(velocity, 2),
                "cond_velocity_drop": velocity > 35.0,
                "torso_angle_deg": round(angle, 1),
                "cond_torso_tilt": angle >= 55.0,
                "aspect_ratio": round(aspect_ratio, 2),
                "cond_aspect_ratio": aspect_ratio >= 1.05,
                "state": state,
                "fall_alert": is_alert
            })

        if has_fall:
            fall_events.append(FallEvent(
                timestamp=f"{int(fall_start_sec // 60):02d}:{fall_start_sec % 60:05.2f}",
                time_sec=round(fall_start_sec, 2),
                frame_idx=fall_start_frame,
                track_id=1,
                event_type="Sudden Fall Confirmed",
                severity="HIGH",
                confidence=0.92,
                torso_angle=76.4,
                vertical_velocity=44.8,
                aspect_ratio=1.48,
                condition_triggers=["Descent Velocity > 35px", "Torso Angle > 55°", "Aspect Ratio > 1.05"]
            ))

        # Generate a lightweight annotated video output
        annotated_path = self._generate_mock_annotated_video(
            video_path, fall_start_frame, total_frames, fps, width, height
        )

        proc_time = round(time.time() - start_time, 2)

        return AnalysisResult(
            video_name=os.path.basename(video_path),
            video_path=video_path,
            duration_sec=round(duration_sec, 2),
            fps=round(fps, 2),
            total_frames=total_frames,
            resolution=f"{width}x{height}",
            persons_tracked=1,
            risk_level="CRITICAL" if has_fall else "NORMAL",
            fall_detected=has_fall,
            fall_events=fall_events,
            telemetry_data=telemetry,
            annotated_video_path=annotated_path,
            processing_time_sec=proc_time,
            model_name="SafeCCTV Heuristic Simulator (Mock Mode)"
        )

    def _generate_mock_annotated_video(
        self,
        source_path: str,
        fall_frame: int,
        total_frames: int,
        fps: float,
        width: int,
        height: int
    ) -> Optional[str]:
        """
        Creates a lightweight annotated preview video with simulated bounding boxes,
        skeleton lines, and SafeCCTV HUD overlay.
        """
        out_dir = os.path.dirname(source_path) or "."
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_mock_annotated.mp4")

        try:
            cap = cv2.VideoCapture(source_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            f_idx = 0
            # Limit to 180 frames to keep mock generation instant and lightweight
            max_preview_frames = min(180, total_frames) if total_frames > 0 else 90

            while cap.isOpened() and f_idx < max_preview_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Determine simulated state
                is_fall_active = abs(f_idx - fall_frame) < int(fps * 1.5)
                is_descent = abs(f_idx - fall_frame) < int(fps * 0.5)

                if is_fall_active:
                    color = (0, 0, 255)  # Red
                    status_text = "🚨 POSSIBLE FALL DETECTED"
                    box_coords = (int(width * 0.35), int(height * 0.65), int(width * 0.70), int(height * 0.90))
                elif is_descent:
                    color = (0, 200, 255)  # Yellow
                    status_text = "⚠️ RAPID DESCENT"
                    box_coords = (int(width * 0.40), int(height * 0.45), int(width * 0.62), int(height * 0.85))
                else:
                    color = (0, 255, 0)  # Green
                    status_text = "NORMAL UPRIGHT"
                    box_coords = (int(width * 0.42), int(height * 0.25), int(width * 0.58), int(height * 0.85))

                # Draw simulated bounding box
                x1, y1, x2, y2 = box_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:1 | {status_text}", (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                # SafeCCTV Master HUD
                t_sec = f_idx / fps
                ts = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"
                cv2.rectangle(frame, (0, 0), (width, 36), (20, 20, 20), -1)
                cv2.putText(frame, f"SafeCCTV [MOCK ANNOTATION] | {ts} | Status: {status_text}",
                            (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                out.write(frame)
                f_idx += 1

            cap.release()
            out.release()
            return out_path
        except Exception:
            # Fallback to source video if OpenCV writing encounters format issues
            return source_path


def get_detector(
    use_real_model: bool = True,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    simulate_fall: bool = True
) -> BaseFallDetector:
    """
    Factory function for instantiating video fall detectors:
    - use_real_model=True: Returns MobileNetV2LSTMVideoDetector using the trained .keras weights.
    - use_real_model=False: Returns MockFallDetector heuristic simulation.
    """
    if use_real_model:
        return MobileNetV2LSTMVideoDetector(model_path=model_path, threshold=threshold)
    return MockFallDetector(simulate_fall=simulate_fall)


# -----------------------------------------------------------------------------
# REAL MOBILENETV2-LSTM VIDEO DETECTOR IMPLEMENTATION
# -----------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def load_fall_model(model_path: Optional[str] = None) -> Any:
    """
    Loads and caches the trained MobileNetV2-LSTM fall detection model.
    Conforms to the exact deserialization specification:
    tf.keras.models.load_model(model_path, compile=False, custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input})
    """
    if model_path is None:
        model_path = os.environ.get("FALL_MODEL_PATH", "research_model/best_fall_model.keras")

    abs_path = os.path.abspath(model_path)
    with _MODEL_LOCK:
        if abs_path not in _MODEL_CACHE:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Trained MobileNetV2-LSTM model file not found at: {abs_path}")
            import tensorflow as tf
            loaded = tf.keras.models.load_model(
                abs_path,
                compile=False,
                custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
            )
            _MODEL_CACHE[abs_path] = loaded
        return _MODEL_CACHE[abs_path]


def preprocess_frame(bgr_frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses a single video frame to match training:
    - Decodes OpenCV BGR frame and converts to RGB
    - Resizes to 224x224 using cv2.INTER_AREA
    - Preserves pixel values in the [0.0, 255.0] range as float32
      (the model's internal mobilenetv2_preprocess Lambda layer normalizes to [-1, 1]).
    """
    resized = cv2.resize(bgr_frame, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def extract_sampled_sequences(
    video_path: str,
    sampling_fps: float = 3.0,
    sequence_length: int = 15,
    stride: int = 5
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts 15-frame sequences sampled at 3 FPS from a video file.
    Matches research_model.dataset.VideoSequenceDataset:
    - Downsamples video frames to sampling_fps (3.0 FPS).
    - Resizes frames to 224x224 RGB float32 [0, 255].
    - Forms sliding windows of sequence_length (15) with step stride (5).
    - Handles short videos (< 15 sampled frames) by repeating the last frame
      (exact match to research_model/dataset.py:382).
    - Returns: (sequences_array, window_metadata_list, video_info_dict)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file at: {video_path}")

    try:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / orig_fps if total_frames > 0 else 0.0

        video_info = {
            "fps": float(orig_fps),
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(duration_sec)
        }

        step = max(1, int(round(orig_fps / sampling_fps)))
        sampled_frames = []
        sampled_meta = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                t_sec = frame_idx / orig_fps
                sampled_frames.append(preprocess_frame(frame))
                sampled_meta.append({
                    "sampled_idx": len(sampled_frames) - 1,
                    "orig_frame_idx": frame_idx,
                    "time_sec": t_sec,
                    "timestamp": f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"
                })

            frame_idx += 1
    finally:
        cap.release()

    total_sampled = len(sampled_frames)
    if total_sampled == 0:
        empty_seq = np.empty((0, sequence_length, 224, 224, 3), dtype=np.float32)
        return empty_seq, [], video_info

    sequences = []
    window_metadata = []

    if total_sampled < sequence_length:
        # Video shorter than sequence_length (e.g. < 5.0s):
        # Match training/evaluation padding strategy (dataset.py:382)
        pad_count = sequence_length - total_sampled
        last_frame = sampled_frames[-1]
        padded = sampled_frames + [last_frame] * pad_count
        sequences.append(padded)
        window_metadata.append({
            "window_idx": 0,
            "start_sampled_idx": 0,
            "end_sampled_idx": total_sampled - 1,
            "start_time_sec": sampled_meta[0]["time_sec"],
            "end_time_sec": sampled_meta[-1]["time_sec"],
            "center_frame_idx": sampled_meta[total_sampled // 2]["orig_frame_idx"],
            "is_padded": True,
            "original_frames_count": total_sampled
        })
    else:
        # Sliding window with step = stride
        for start_idx in range(0, total_sampled - sequence_length + 1, stride):
            window = sampled_frames[start_idx : start_idx + sequence_length]
            sequences.append(window)
            mid_offset = sequence_length // 2
            window_metadata.append({
                "window_idx": len(sequences) - 1,
                "start_sampled_idx": start_idx,
                "end_sampled_idx": start_idx + sequence_length - 1,
                "start_time_sec": sampled_meta[start_idx]["time_sec"],
                "end_time_sec": sampled_meta[start_idx + sequence_length - 1]["time_sec"],
                "center_frame_idx": sampled_meta[start_idx + mid_offset]["orig_frame_idx"],
                "is_padded": False,
                "original_frames_count": sequence_length
            })

        # Ensure final tail frames are included if not aligned with stride (matching dataset.py:391)
        if (total_sampled - sequence_length) % stride != 0:
            start_idx = total_sampled - sequence_length
            window = sampled_frames[start_idx:]
            sequences.append(window)
            mid_offset = sequence_length // 2
            window_metadata.append({
                "window_idx": len(sequences) - 1,
                "start_sampled_idx": start_idx,
                "end_sampled_idx": total_sampled - 1,
                "start_time_sec": sampled_meta[start_idx]["time_sec"],
                "end_time_sec": sampled_meta[-1]["time_sec"],
                "center_frame_idx": sampled_meta[start_idx + mid_offset]["orig_frame_idx"],
                "is_padded": False,
                "original_frames_count": sequence_length
            })

    return np.array(sequences, dtype=np.float32), window_metadata, video_info


class MobileNetV2LSTMVideoDetector(BaseFallDetector):
    """
    Production video fall detector powered by the trained MobileNetV2-LSTM neural network.
    Extracts 224x224 RGB frames at 3.0 FPS, groups them into 15-frame sliding window sequences
    (stride 5), and performs batch inference with configurable decision threshold.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        sampling_fps: float = 3.0,
        sequence_length: int = 15,
        stride: int = 5
    ):
        self.model_path = model_path if model_path is not None else os.environ.get(
            "FALL_MODEL_PATH", "research_model/best_fall_model.keras"
        )
        self.threshold = float(threshold)
        self.sampling_fps = float(sampling_fps)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)

    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> AnalysisResult:
        start_time = time.time()

        if progress_callback:
            progress_callback(0.05, "Validating input video and decoding stream...")

        # 1. Extract 15-frame sliding sequences sampled at 3 FPS
        sequences, window_meta, v_info = extract_sampled_sequences(
            video_path=video_path,
            sampling_fps=self.sampling_fps,
            sequence_length=self.sequence_length,
            stride=self.stride
        )

        total_frames = v_info["total_frames"]
        fps = v_info["fps"]
        width = v_info["width"]
        height = v_info["height"]
        duration_sec = v_info["duration_sec"]

        if progress_callback:
            progress_callback(0.25, f"Extracted {len(sequences)} temporal sequence windows at {self.sampling_fps} FPS...")

        # 2. Check for empty video / zero decodable frames
        if len(sequences) == 0:
            if progress_callback:
                progress_callback(1.0, "Analysis complete (no decodable frames).")
            return AnalysisResult(
                video_name=os.path.basename(video_path),
                video_path=video_path,
                duration_sec=round(duration_sec, 2),
                fps=round(fps, 2),
                total_frames=total_frames,
                resolution=f"{width}x{height}",
                persons_tracked=0,
                risk_level="NORMAL",
                fall_detected=False,
                fall_events=[],
                telemetry_data=[],
                annotated_video_path=None,
                processing_time_sec=round(time.time() - start_time, 2),
                model_name="MobileNetV2-LSTM (Zero sequences extracted)"
            )

        # 3. Load model from cache and run inference
        if progress_callback:
            progress_callback(0.40, "Loading MobileNetV2-LSTM model weights...")

        model = load_fall_model(self.model_path)

        if progress_callback:
            progress_callback(0.55, f"Running deep learning inference on {len(sequences)} sequence windows...")

        # Run forward pass (model outputs probabilities in [0.0, 1.0])
        probs = model(sequences, training=False).numpy().flatten()

        if progress_callback:
            progress_callback(0.75, "Compiling fall events and telemetry data...")

        # 4. Formulate FallEvents and Telemetry
        fall_events = []
        telemetry = []
        has_fall = False

        for k, prob in enumerate(probs):
            prob = float(prob)
            w_info = window_meta[k]
            start_sec = w_info["start_time_sec"]
            end_sec = w_info["end_time_sec"]
            center_frame = w_info["center_frame_idx"]
            timestamp_str = f"{int(start_sec // 60):02d}:{start_sec % 60:05.2f}"
            is_seq_fall = (prob >= self.threshold)

            if is_seq_fall:
                has_fall = True
                if prob >= 0.80:
                    severity = "CRITICAL"
                elif prob >= 0.65:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                triggers = [f"MobileNetV2-LSTM Prob: {prob * 100:.1f}% (Threshold: {self.threshold * 100:.0f}%)"]
                if w_info.get("is_padded", False):
                    triggers.append(f"Short video padded ({w_info.get('original_frames_count', 0)} to 15 frames)")

                fall_events.append(FallEvent(
                    timestamp=timestamp_str,
                    time_sec=round(start_sec, 2),
                    frame_idx=center_frame,
                    track_id=1,
                    event_type="Fall Confirmed (MobileNetV2-LSTM)",
                    severity=severity,
                    confidence=round(prob, 4),
                    torso_angle=0.0,
                    vertical_velocity=0.0,
                    aspect_ratio=0.0,
                    condition_triggers=triggers
                ))

            telemetry.append({
                "frame": center_frame,
                "timestamp": timestamp_str,
                "time_sec": round(start_sec, 2),
                "window_start_sec": round(start_sec, 2),
                "window_end_sec": round(end_sec, 2),
                "track_id": 1,
                "fall_probability": round(prob, 4),
                "confidence": round(prob, 4),
                "threshold": self.threshold,
                "fall_alert": is_seq_fall,
                "state": "FALL_ALERT" if is_seq_fall else "NORMAL"
            })

        # 5. Generate Annotated Preview Video
        if progress_callback:
            progress_callback(0.85, "Synthesizing annotated overlay preview video...")

        annotated_path = self._generate_annotated_video(
            video_path=video_path,
            window_meta=window_meta,
            probs=probs,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames
        )

        proc_time = round(time.time() - start_time, 2)
        if progress_callback:
            progress_callback(1.0, f"Analysis complete in {proc_time}s.")

        return AnalysisResult(
            video_name=os.path.basename(video_path),
            video_path=video_path,
            duration_sec=round(duration_sec, 2),
            fps=round(fps, 2),
            total_frames=total_frames,
            resolution=f"{width}x{height}",
            persons_tracked=1 if len(sequences) > 0 else 0,
            risk_level="CRITICAL" if has_fall else "NORMAL",
            fall_detected=has_fall,
            fall_events=fall_events,
            telemetry_data=telemetry,
            annotated_video_path=annotated_path,
            processing_time_sec=proc_time,
            model_name="MobileNetV2-LSTM Spatio-Temporal Fall Detector"
        )

    def _generate_annotated_video(
        self,
        video_path: str,
        window_meta: List[Dict[str, Any]],
        probs: np.ndarray,
        fps: float,
        width: int,
        height: int,
        total_frames: int
    ) -> Optional[str]:
        """
        Creates an annotated preview video with safe HUD overlay displaying the
        MobileNetV2-LSTM fall detection probability and state.
        """
        out_dir = os.path.dirname(video_path) or "."
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_mobilenet_annotated.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path

        out = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not out.isOpened():
                return video_path

            max_preview_frames = min(300, total_frames) if total_frames > 0 else 150
            f_idx = 0

            while cap.isOpened() and f_idx < max_preview_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                t_sec = f_idx / fps
                ts = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"

                # Find all window probabilities covering this timestamp
                active_probs = [
                    probs[i] for i, w in enumerate(window_meta)
                    if w["start_time_sec"] <= t_sec <= (w["end_time_sec"] + 0.5)
                ]
                current_prob = float(max(active_probs)) if active_probs else (float(probs[0]) if len(probs) > 0 else 0.0)
                is_alert = (current_prob >= self.threshold)

                # Visual HUD banner
                if is_alert:
                    bg_color = (0, 0, 180)  # Red
                    status_text = f"CRITICAL: FALL DETECTED ({current_prob * 100:.1f}%)"
                    border_color = (0, 0, 255)
                else:
                    bg_color = (20, 20, 20)  # Dark
                    status_text = f"NORMAL ({current_prob * 100:.1f}%)"
                    border_color = (0, 255, 0)

                # Draw Top HUD
                cv2.rectangle(frame, (0, 0), (width, 36), bg_color, -1)
                hud_msg = f"SafeCCTV [MobileNetV2-LSTM] | {ts} | Status: {status_text} | Thresh: {self.threshold:.2f}"
                cv2.putText(frame, hud_msg, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

                # If alert, draw alert border
                if is_alert:
                    cv2.rectangle(frame, (4, 4), (width - 4, height - 4), border_color, 3)

                out.write(frame)
                f_idx += 1

            return out_path
        except Exception:
            return video_path
        finally:
            cap.release()
            if out is not None:
                out.release()

