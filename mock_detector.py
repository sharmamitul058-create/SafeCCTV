"""
SafeCCTV - Modular Detector Interface & Mock Detection Service
Designed for clean separation between Streamlit UI and Computer Vision detector engines.
"""

import os
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable


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


def get_detector(use_real_model: bool = False) -> BaseFallDetector:
    """
    Factory function for easy swapping between Mock and Real YOLO detector in future.
    """
    if use_real_model:
        raise NotImplementedError("Real detector integration will be attached here.")
    return MockFallDetector()
