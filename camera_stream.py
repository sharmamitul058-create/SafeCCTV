"""
SafeCCTV - Background Camera Stream & Incident Manager
Handles OpenCV VideoCapture (Webcam, RTSP CCTV, and Labeled Test Footage),
thread synchronization, frozen feed detection, alarm latching, incident evidence saving,
and credential sanitization.
"""

import os
import re
import cv2
import time
import winsound
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List
from fall_detector import RealtimeFallDetector


def sanitize_camera_source(source: Any) -> str:
    """Masks passwords from RTSP URLs to prevent credential leakage in logs and UI."""
    if isinstance(source, int):
        return f"Webcam Device #{source}"
    src_str = str(source)
    # Mask rtsp://user:password@host:port -> rtsp://user:***@host:port
    sanitized = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", src_str)
    return sanitized


class CameraStreamManager:
    """
    Thread-safe camera worker that captures frames from Webcams or RTSP CCTV,
    executes real-time fall detection, manages alarm states, and logs incidents.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraStreamManager()
            return cls._instance

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.connection_status = "DISCONNECTED"  # DISCONNECTED, CONNECTING, LIVE, CAMERA ERROR, FROZEN
        self.error_message = ""
        self.source_label = "Webcam Device #0"
        self.raw_source = 0
        self.is_prerecorded = False
        self.start_time = None

        self.capture_thread = None
        self.inference_thread = None
        self.lstm_thread = None

        self.frame_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.raw_frame_id = 0
        self.processed_frame_id = 0
        self.last_frame_time = 0.0
        self.last_capture_time = 0.0
        self.last_inference_time = 0.0
        self.first_capture_received = False
        self.first_frame_received = False
        self.fps = 0.0

        # MobileNetV2-LSTM Fall Detection Pipeline
        self.lstm_model_path = os.environ.get("FALL_MODEL_PATH", "research_model/best_fall_model.keras")
        self.lstm_threshold = float(os.environ.get("FALL_THRESHOLD", 0.50))
        self.lstm_model = None
        self.sequence_buffer = deque(maxlen=15)
        self.buffer_lock = threading.Lock()
        self.latest_fall_prob = 0.0
        self.latest_lstm_status = "INITIALIZING"
        self.lstm_alert_latched = False
        self.lstm_inference_count = 0
        self.last_lstm_time = 0.0
        self.last_sample_time = 0.0
        self.sampling_interval = 1.0 / 3.0  # 3 FPS downsampling (~0.333s)

        self.alarm_lock = threading.Lock()
        self.alarm_active = False
        self.alarm_muted = False
        self.alarm_acknowledged = False
        self.last_incident_info = None
        self.incidents_history = []

        # Directory structure
        self.incident_dir = "E:/SafeCCTV/incidents"
        os.makedirs(self.incident_dir, exist_ok=True)
        self.csv_log_path = os.path.join(self.incident_dir, "incident_log.csv")
        self._init_csv_log()

        # Initialize detector
        self.detector = RealtimeFallDetector(model_path="yolov8n-pose.pt")

        # Audio thread (sounds ONLY when alarm_active is True and not muted)
        self.audio_running = True
        self.audio_thread = threading.Thread(target=self._audio_alarm_loop, daemon=True)
        self.audio_thread.start()

    def _init_csv_log(self):
        """Initializes incident CSV log with header if not present."""
        if not os.path.exists(self.csv_log_path) or os.path.getsize(self.csv_log_path) == 0:
            df = pd.DataFrame(columns=[
                "timestamp", "fall_time_sec", "camera_id", "track_id",
                "severity", "torso_angle", "descent_velocity", "aspect_ratio",
                "confidence", "snapshot_file", "status"
            ])
            df.to_csv(self.csv_log_path, index=False)
        else:
            try:
                df = pd.read_csv(self.csv_log_path)
                self.incidents_history = df.to_dict("records")
            except Exception:
                self.incidents_history = []

    def connect(self, source=0, is_prerecorded: bool = False, source_name: Optional[str] = None):
        """
        Connects to video source (Webcam, RTSP URL, or test clip) and starts streaming workers.
        """
        if self.is_running:
            self.disconnect()

        self.raw_source = source
        self.is_prerecorded = is_prerecorded
        self.source_label = source_name if source_name else sanitize_camera_source(source)
        self.connection_status = "CONNECTING"
        self.error_message = ""
        self.start_time = time.time()

        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_annotated_frame = None
            self.raw_frame_id = 0
            self.processed_frame_id = 0
            self.last_frame_time = time.time()
            self.last_capture_time = time.time()
            self.last_inference_time = time.time()
            self.first_capture_received = False
            self.first_frame_received = False
            self.fps = 0.0

        with self.buffer_lock:
            self.sequence_buffer.clear()
        self.latest_fall_prob = 0.0
        self.latest_lstm_status = "INITIALIZING"
        self.lstm_alert_latched = False
        self.last_sample_time = time.time()

        self.is_running = True

        self.capture_thread = threading.Thread(target=self._capture_loop, args=(source,), daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.lstm_thread = threading.Thread(target=self._lstm_worker_loop, daemon=True)

        self.capture_thread.start()
        self.inference_thread.start()
        self.lstm_thread.start()
        return True

    def disconnect(self):
        """Cleanly stops workers and releases camera device without freezing."""
        self.is_running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        self.inference_thread = None

        if self.lstm_thread and self.lstm_thread.is_alive():
            self.lstm_thread.join(timeout=1.0)
        self.lstm_thread = None

        with self.alarm_lock:
            self.alarm_active = False

        self.connection_status = "DISCONNECTED"
        return True

    def _capture_loop(self, source):
        """Continuously reads frames from laptop webcam (device 0)."""
        try:
            device_idx = int(source) if isinstance(source, (int, str)) and str(source).isdigit() else 0
            
            # Use DirectShow backend on Windows with 640x480 resolution
            self.cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(device_idx)

            if not self.cap or not self.cap.isOpened():
                self.connection_status = "CAMERA ERROR"
                self.error_message = f"Failed to open Laptop Webcam (Device #{device_idx}). Please verify camera is connected and not in use by another app."
                self.is_running = False
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up sensor: discard initial unstable black/dark frames
            for _ in range(5):
                self.cap.read()
                time.sleep(0.03)

            consecutive_failures = 0
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()

                # Step 1: Verify frame returned is a valid BGR image with correct shape and uint8 dtype
                if not ret or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= 15:
                        self.connection_status = "CAMERA ERROR"
                        self.error_message = "Laptop webcam read failed repeatedly. Camera may have been disconnected."
                        break
                    time.sleep(0.04)
                    continue

                if frame.dtype != np.uint8 or len(frame.shape) != 3 or frame.shape[2] != 3:
                    # Invalid format
                    consecutive_failures += 1
                    time.sleep(0.04)
                    continue

                consecutive_failures = 0

                # Step 2: Buffer synchronization - store an independent frame copy so readers never share mutable buffers
                clean_copy = frame.copy()

                with self.frame_lock:
                    self.latest_raw_frame = clean_copy
                    self.raw_frame_id += 1
                    self.last_capture_time = time.time()
                    self.first_capture_received = True

        except Exception as e:
            self.connection_status = "CAMERA ERROR"
            self.error_message = str(e)
        finally:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _inference_loop(self):
        """Runs YOLOv8-pose inference and fall detection on latest available frame."""
        fps_counter = 0
        fps_timer = time.time()
        last_seen_id = -1

        while self.is_running:
            target_frame = None
            frame_id = -1

            # Acquire latest frame reference under lock without redundant deep copy
            with self.frame_lock:
                if self.latest_raw_frame is not None and self.raw_frame_id != last_seen_id:
                    target_frame = self.latest_raw_frame
                    frame_id = self.raw_frame_id
                    last_seen_id = frame_id

            if target_frame is None:
                time.sleep(0.002)
                continue

            now = time.time()
            fps_counter += 1
            if now - fps_timer >= 1.0:
                self.fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                # Detector creates an internal copy for drawing annotations; target_frame remains untouched
                annotated, new_events = self.detector.process_frame(target_frame, timestamp_str)
            except Exception:
                annotated = target_frame.copy()
                new_events = []

            # 3 FPS temporal frame sampling for MobileNetV2-LSTM sequence buffer
            now_sample = time.time()
            if (now_sample - self.last_sample_time) >= self.sampling_interval:
                try:
                    resized = cv2.resize(target_frame, (224, 224), interpolation=cv2.INTER_AREA)
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                    with self.buffer_lock:
                        self.sequence_buffer.append(rgb_frame)
                    self.last_sample_time = now_sample
                except Exception:
                    pass

            # Master HUD Overlay with MobileNetV2-LSTM Fall Telemetry
            prob_pct = self.latest_fall_prob * 100.0
            is_fall_active = self.alarm_active or (self.latest_lstm_status == "FALL_ALERT")
            status_text = "ALARM: FALL DETECTED" if is_fall_active else "NORMAL"
            hud_bg = (0, 0, 180) if is_fall_active else (20, 20, 20)

            height, width = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (width, 36), hud_bg, -1)
            hud_text = f"SafeCCTV [MobileNetV2-LSTM] | P(Fall): {prob_pct:.1f}% | {timestamp_str} | Status: {status_text} | {self.fps:.1f} FPS"
            cv2.putText(
                annotated,
                hud_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # Draw alert border if fall is triggered
            if is_fall_active:
                cv2.rectangle(annotated, (4, 4), (width - 4, height - 4), (0, 0, 255), 3)

            # Trigger fall event handling
            if new_events:
                self._handle_fall_events(new_events, target_frame, annotated, timestamp_str)

            with self.frame_lock:
                # annotated is a fresh array from process_frame and unmutated thereafter
                self.latest_annotated_frame = annotated
                self.processed_frame_id = frame_id
                self.last_inference_time = time.time()
                self.last_frame_time = self.last_inference_time
                self.first_frame_received = True

    def _handle_fall_events(self, events, raw_frame, annotated_frame, timestamp_str):
        """Processes confirmed fall event: records fall time, saves snapshot, activates alarm."""
        fall_time_sec = round(time.time() - self.start_time, 2) if self.start_time else 0.0

        for ev in events:
            track_id = ev["track_id"]
            incident_id = f"inc_{int(time.time())}_id{track_id}"
            snapshot_filename = f"{incident_id}.jpg"
            snapshot_path = os.path.join(self.incident_dir, snapshot_filename)

            try:
                cv2.imwrite(snapshot_path, annotated_frame)
            except Exception:
                pass

            incident_record = {
                "timestamp": timestamp_str,
                "fall_time_sec": fall_time_sec,
                "camera_id": self.source_label,
                "track_id": track_id,
                "severity": "CRITICAL",
                "torso_angle": round(ev.get("torso_angle", 0.0), 1),
                "descent_velocity": round(ev.get("descent_velocity", 0.0), 1),
                "aspect_ratio": round(ev.get("aspect_ratio", 0.0), 2),
                "confidence": round(ev.get("confidence", 0.8), 2),
                "snapshot_file": snapshot_filename,
                "status": "UNACKNOWLEDGED"
            }

            self.last_incident_info = incident_record
            self.incidents_history.insert(0, incident_record)

            # Append to persistent CSV
            try:
                df_new = pd.DataFrame([incident_record])
                df_new.to_csv(self.csv_log_path, mode="a", header=False, index=False)
            except Exception:
                pass

            # Activate audible alarm (triggers only once until acknowledged or reset)
            with self.alarm_lock:
                self.alarm_active = True
                self.alarm_muted = False
                self.alarm_acknowledged = False

    def _lstm_worker_loop(self):
        """
        Dedicated background worker for MobileNetV2-LSTM Spatio-Temporal inference.
        Evaluates 15-frame rolling temporal sequences (sampled at 3 FPS) without blocking
        the camera capture thread or Streamlit UI rendering.
        """
        model = self.lstm_model
        if model is None:
            try:
                from mock_detector import load_fall_model
                model = load_fall_model(self.lstm_model_path)
                self.lstm_model = model
            except Exception as e:
                self.latest_lstm_status = f"MODEL ERROR: {e}"
                return

        while self.is_running:
            frames_snapshot = None
            with self.buffer_lock:
                buf_len = len(self.sequence_buffer)
                if buf_len > 0:
                    if buf_len < 15:
                        # Pad short sequences by repeating last frame (matching training pipeline)
                        pad_count = 15 - buf_len
                        last_f = self.sequence_buffer[-1]
                        frames_snapshot = list(self.sequence_buffer) + [last_f] * pad_count
                    else:
                        frames_snapshot = list(self.sequence_buffer)

            if frames_snapshot is None:
                time.sleep(0.04)
                continue

            try:
                seq_tensor = np.expand_dims(np.array(frames_snapshot, dtype=np.float32), axis=0)
                # Model inference forward pass
                prob = float(model(seq_tensor, training=False)[0, 0])
                self.latest_fall_prob = prob
                self.last_lstm_time = time.time()
                self.lstm_inference_count += 1

                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if prob >= self.lstm_threshold:
                    self.latest_lstm_status = "FALL_ALERT"
                    if not self.lstm_alert_latched:
                        self.lstm_alert_latched = True
                        with self.frame_lock:
                            raw_f = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None
                            ann_f = self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else raw_f
                        if raw_f is not None and ann_f is not None:
                            self._handle_lstm_fall_event(prob, raw_f, ann_f, timestamp_str)
                else:
                    if prob < max(0.0, self.lstm_threshold - 0.10):
                        self.latest_lstm_status = "NORMAL"
                        self.lstm_alert_latched = False
            except Exception:
                pass

            time.sleep(0.04)

    def _handle_lstm_fall_event(self, prob: float, raw_frame, annotated_frame, timestamp_str):
        """Processes confirmed MobileNetV2-LSTM fall event: saves evidence snapshot, logs incident, triggers alarm."""
        fall_time_sec = round(time.time() - self.start_time, 2) if self.start_time else 0.0
        incident_id = f"inc_{int(time.time())}_lstm"
        snapshot_filename = f"{incident_id}.jpg"
        snapshot_path = os.path.join(self.incident_dir, snapshot_filename)

        try:
            cv2.imwrite(snapshot_path, annotated_frame)
        except Exception:
            pass

        incident_record = {
            "timestamp": timestamp_str,
            "fall_time_sec": fall_time_sec,
            "camera_id": self.source_label,
            "track_id": 1,
            "severity": "CRITICAL" if prob >= 0.80 else "HIGH" if prob >= 0.65 else "MEDIUM",
            "torso_angle": 0.0,
            "descent_velocity": 0.0,
            "aspect_ratio": 0.0,
            "confidence": round(prob, 4),
            "snapshot_file": snapshot_filename,
            "status": "UNACKNOWLEDGED"
        }

        self.last_incident_info = incident_record
        self.incidents_history.insert(0, incident_record)

        try:
            df_new = pd.DataFrame([incident_record])
            df_new.to_csv(self.csv_log_path, mode="a", header=False, index=False)
        except Exception:
            pass

        with self.alarm_lock:
            self.alarm_active = True
            self.alarm_muted = False
            self.alarm_acknowledged = False

    def get_lstm_metrics(self) -> Dict[str, Any]:
        """Returns thread-safe MobileNetV2-LSTM telemetry and status."""
        return {
            "latest_fall_prob": self.latest_fall_prob,
            "latest_lstm_status": self.latest_lstm_status,
            "lstm_threshold": self.lstm_threshold,
            "lstm_inference_count": self.lstm_inference_count,
            "buffer_size": len(self.sequence_buffer),
            "alert_latched": self.lstm_alert_latched,
            "last_lstm_time": self.last_lstm_time
        }

    def get_latest_frame(self, return_raw: bool = False):
        """Returns the latest frame (annotated or raw), connection status, fps, alarm status, and elapsed seconds."""
        with self.frame_lock:
            now = time.time()
            if self.is_running and self.first_capture_received:
                # Track camera capture health separately from YOLO inference.
                # Only mark FROZEN if camera capture itself stops delivering frames (>3.5s).
                # Slow YOLO inference does not mark a working camera as frozen.
                if (now - self.last_capture_time) > 3.5:
                    self.connection_status = "FROZEN"
                else:
                    self.connection_status = "LIVE"
            elif self.is_running and not self.first_capture_received:
                if self.connection_status != "CAMERA ERROR":
                    self.connection_status = "CONNECTING"

            target = self.latest_raw_frame if return_raw else self.latest_annotated_frame
            if target is None and self.latest_raw_frame is not None:
                target = self.latest_raw_frame

            frame = target
            elapsed_sec = round(now - self.start_time, 1) if (self.start_time and self.is_running) else 0.0

            return frame, self.connection_status, self.fps, self.alarm_active, elapsed_sec

    def get_health_status(self):
        """Returns thread-safe capture vs inference health metrics and timestamps."""
        with self.frame_lock:
            now = time.time()
            capture_age = (now - self.last_capture_time) if self.last_capture_time > 0 else None
            inference_age = (now - self.last_inference_time) if self.last_inference_time > 0 else None

            capture_healthy = (
                self.is_running
                and self.first_capture_received
                and capture_age is not None
                and capture_age <= 3.5
            )
            inference_healthy = (
                self.is_running
                and self.first_frame_received
                and inference_age is not None
                and inference_age <= 3.5
            )

            return {
                "connection_status": self.connection_status,
                "capture_healthy": capture_healthy,
                "inference_healthy": inference_healthy,
                "last_capture_time": self.last_capture_time,
                "last_inference_time": self.last_inference_time,
                "capture_age_sec": round(capture_age, 3) if capture_age is not None else None,
                "inference_age_sec": round(inference_age, 3) if inference_age is not None else None,
                "raw_frame_id": self.raw_frame_id,
                "processed_frame_id": self.processed_frame_id,
                "fps": self.fps,
            }

    def _audio_alarm_loop(self):
        """Audible Windows alarm loop that beeps ONLY when a fall is confirmed and not muted."""
        while self.audio_running:
            should_sound = False
            with self.alarm_lock:
                if self.alarm_active and not self.alarm_muted and not self.alarm_acknowledged:
                    should_sound = True

            if should_sound:
                try:
                    winsound.Beep(950, 220)
                    winsound.Beep(650, 220)
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                time.sleep(0.08)

    def acknowledge_alarm(self):
        """Acknowledges the alert: silences the audio alarm and updates incident status."""
        with self.alarm_lock:
            self.alarm_acknowledged = True
            self.alarm_muted = True
            self.alarm_active = False

        if self.detector:
            self.detector.acknowledge_tracker_alerts()

        # Latch ongoing LSTM alert so it does not re-trigger while person remains down
        self.lstm_alert_latched = True

        if self.last_incident_info:
            self.last_incident_info["status"] = "ACKNOWLEDGED"

    def mute_alarm(self):
        """Mutes audio siren temporarily while keeping the visual alert state intact."""
        with self.alarm_lock:
            self.alarm_muted = True

    def reset_alarm(self):
        """Completely resets alarm and detection state back to normal monitoring."""
        with self.alarm_lock:
            self.alarm_active = False
            self.alarm_muted = False
            self.alarm_acknowledged = False

        if self.detector:
            self.detector.reset_tracker_alerts()

        # Reset LSTM latch and status
        self.lstm_alert_latched = False
        self.latest_lstm_status = "NORMAL"

        if self.last_incident_info:
            self.last_incident_info["status"] = "SYSTEM_RESET"

    def clear_runtime_data(self) -> Dict[str, Any]:
        """
        Safely stops the camera and alarm, wipes old snapshots and CSV logs,
        and clears all in-memory caches, trackers, and counters for a fresh test.
        Preserves source code, weights, and configurations.
        """
        # 1. Safely stop camera and alarm first
        self.disconnect()

        with self.alarm_lock:
            self.alarm_active = False
            self.alarm_muted = False
            self.alarm_acknowledged = False
            self.last_incident_info = None
            self.incidents_history.clear()

        # Reset MobileNetV2-LSTM sequence buffer and status
        with self.buffer_lock:
            self.sequence_buffer.clear()
        self.latest_fall_prob = 0.0
        self.latest_lstm_status = "INITIALIZING"
        self.lstm_alert_latched = False
        self.lstm_inference_count = 0

        # 2. Reset frame buffers and counters under frame_lock
        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_annotated_frame = None
            self.raw_frame_id = 0
            self.processed_frame_id = 0
            self.last_frame_time = 0.0
            self.last_capture_time = 0.0
            self.last_inference_time = 0.0
            self.first_capture_received = False
            self.first_frame_received = False
            self.fps = 0.0

        # 3. Reset fall detector tracker and dynamics
        if self.detector:
            self.detector.clear_all_tracks()

        # 4. Safely delete snapshot files in incidents directory (only .jpg files)
        deleted_snapshots = 0
        if os.path.exists(self.incident_dir):
            for filename in os.listdir(self.incident_dir):
                if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                    filepath = os.path.join(self.incident_dir, filename)
                    try:
                        os.remove(filepath)
                        deleted_snapshots += 1
                    except Exception:
                        pass

        # 5. Reset CSV log with clean header
        df_empty = pd.DataFrame(columns=[
            "timestamp", "fall_time_sec", "camera_id", "track_id",
            "severity", "torso_angle", "descent_velocity", "aspect_ratio",
            "confidence", "snapshot_file", "status"
        ])
        df_empty.to_csv(self.csv_log_path, index=False)

        # 6. Delete any temporary scratch images in project root
        for temp_img in ["fresh_frame.jpg", "test_frame.jpg", "warmup_frame.jpg"]:
            temp_path = os.path.join("E:/SafeCCTV", temp_img)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        self.start_time = None
        self.connection_status = "DISCONNECTED"

        return {
            "deleted_snapshots": deleted_snapshots,
            "csv_reset": True,
            "status": "CLEARED"
        }
