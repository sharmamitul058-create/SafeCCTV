"""
SafeCCTV - Real-Time YOLOv8-Pose Fall Detection Engine
Handles multi-person tracking, keypoint validation, motion/posture heuristics, and visual overlays.
"""

import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# Standard COCO 17-Keypoint Skeletal Connections
SKELETON_PAIRS = (
    (5, 6), (11, 12), (5, 11), (6, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
)


class PersonTrackHistory:
    """Maintains posture and vertical motion history for a single tracked person."""
    def __init__(self, track_id: int, max_history: int = 30):
        self.track_id = track_id
        self.history = deque(maxlen=max_history)
        self.state = "NORMAL"         # 'NORMAL', 'RAPID_DESCENT', 'FALL_ALERT', 'LYING_DOWN'
        self.frames_in_horizontal = 0
        self.alert_latched = False
        self.fall_timestamp = None
        self.recent_descents = []
        self.last_seen_frame = 0

    def update(self, frame_idx: int, center_y: float, box_w: float, box_h: float, torso_angle: float, is_valid: bool):
        self.last_seen_frame = frame_idx
        self.history.append({
            "frame": frame_idx,
            "center_y": center_y,
            "w": box_w,
            "h": box_h,
            "angle": torso_angle,
            "valid": is_valid
        })


class RealtimeFallDetector:
    """
    Continuous frame-by-frame YOLOv8 Pose Fall Detector.
    Runs on live webcam or RTSP CCTV streams.
    """
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        velocity_threshold: float = 20.0,
        torso_angle_threshold: float = 55.0,
        aspect_ratio_threshold: float = 1.05,
        window_size: int = 6,
        resting_frames_req: int = 3
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Pre-warmup to eliminate latency on first real frame
        try:
            self.model.track(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        except Exception:
            pass

        self.velocity_threshold = float(velocity_threshold)
        self.torso_angle_threshold = float(torso_angle_threshold)
        self.aspect_ratio_threshold = float(aspect_ratio_threshold)
        self.window_size = int(window_size)
        self.resting_frames_req = int(resting_frames_req)

        self.trackers = {}
        self.frame_count = 0
        self.recent_descents = []

    def extract_features(self, keypoints, box, min_conf: float = 0.4):
        """Extracts torso tilt angle, center of mass, and dimensions safely."""
        x1, y1, x2, y2 = box
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))
        aspect_ratio = box_w / box_h

        # Guard against missing or malformed keypoint arrays
        if keypoints is None or len(keypoints) < 13:
            center_y = float((y1 + y2) / 2.0)
            return center_y, box_w, box_h, aspect_ratio, None, False

        # COCO Keypoints: Left Shoulder (5), Right Shoulder (6), Left Hip (11), Right Hip (12)
        l_sh = keypoints[5]
        r_sh = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]

        valid_shoulders = []
        if len(l_sh) > 2 and float(l_sh[2]) >= min_conf:
            valid_shoulders.append((float(l_sh[0]), float(l_sh[1])))
        if len(r_sh) > 2 and float(r_sh[2]) >= min_conf:
            valid_shoulders.append((float(r_sh[0]), float(r_sh[1])))

        valid_hips = []
        if len(l_hip) > 2 and float(l_hip[2]) >= min_conf:
            valid_hips.append((float(l_hip[0]), float(l_hip[1])))
        if len(r_hip) > 2 and float(r_hip[2]) >= min_conf:
            valid_hips.append((float(r_hip[0]), float(r_hip[1])))

        if not valid_shoulders or not valid_hips:
            center_y = float((y1 + y2) / 2.0)
            return center_y, box_w, box_h, aspect_ratio, None, False

        sh_x = sum(pt[0] for pt in valid_shoulders) / len(valid_shoulders)
        sh_y = sum(pt[1] for pt in valid_shoulders) / len(valid_shoulders)
        hip_x = sum(pt[0] for pt in valid_hips) / len(valid_hips)
        hip_y = sum(pt[1] for pt in valid_hips) / len(valid_hips)

        center_y = hip_y
        dx = abs(sh_x - hip_x)
        dy = abs(sh_y - hip_y)

        if dy < 1e-4:
            torso_angle = 90.0
        else:
            torso_angle = float(np.degrees(np.arctan(dx / dy)))

        return center_y, box_w, box_h, aspect_ratio, torso_angle, True

    def _evaluate_tracker(self, tracker: PersonTrackHistory):
        """Evaluates motion dynamics and posture state transitions."""
        if len(tracker.history) < self.window_size:
            return False, "INITIALIZING", {"velocity": 0.0, "torso_angle": 0.0, "aspect_ratio": 0.0}

        latest = tracker.history[-1]
        baseline = tracker.history[-self.window_size]

        downward_velocity = float(latest["center_y"] - baseline["center_y"])
        aspect_ratio = float(latest["w"] / max(1.0, latest["h"]))
        torso_angle = float(latest["angle"]) if latest["angle"] is not None else None
        is_pose_valid = bool(latest.get("valid", False)) and (torso_angle is not None)

        # Determine horizontal/prone posture:
        # Avoid treating upright sitting (angle < 35°) as horizontal, even if bounding box is wide.
        # When pose data is invalid (missing or low confidence keypoints), do NOT assume horizontal
        # posture based only on bounding-box aspect ratio, preventing false fall alerts.
        if is_pose_valid:
            is_horizontal = (torso_angle >= self.torso_angle_threshold) or (
                aspect_ratio >= self.aspect_ratio_threshold and torso_angle >= 40.0
            )
        else:
            is_horizontal = False

        alert_triggered = False
        status_label = tracker.state

        # Prune descents older than 45 frames (~1.5s) specific to this person
        tracker.recent_descents = [d for d in tracker.recent_descents if (self.frame_count - d["frame"]) < 45]
        self.recent_descents = [d for d in self.recent_descents if (self.frame_count - d["frame"]) < 45]

        if downward_velocity > self.velocity_threshold:
            descent_entry = {"frame": self.frame_count, "y": latest["center_y"]}
            tracker.recent_descents.append(descent_entry)
            self.recent_descents.append(descent_entry)

        # Person-specific recent descent check:
        # Ensures one person's rapid movement cannot influence another person's fall evaluation.
        has_recent_descent = (tracker.state == "RAPID_DESCENT") or (len(tracker.recent_descents) > 0)

        if tracker.state == "NORMAL":
            if downward_velocity > self.velocity_threshold:
                tracker.state = "RAPID_DESCENT"
                status_label = "RAPID_DESCENT"
            elif is_horizontal and has_recent_descent:
                tracker.state = "RAPID_DESCENT"
                tracker.frames_in_horizontal += 1
                status_label = "RAPID_DESCENT"
            elif is_horizontal:
                tracker.state = "LYING_DOWN"
                status_label = "LYING_DOWN"

        elif tracker.state == "RAPID_DESCENT":
            if is_horizontal:
                tracker.frames_in_horizontal += 1
                if tracker.frames_in_horizontal >= self.resting_frames_req:
                    tracker.state = "FALL_ALERT"
                    alert_triggered = True
                    status_label = "FALL_ALERT"
            else:
                if downward_velocity < 0 or (is_pose_valid and torso_angle < 35.0):
                    tracker.state = "NORMAL"
                    tracker.frames_in_horizontal = 0
                    tracker.alert_latched = False

        elif tracker.state == "FALL_ALERT":
            alert_triggered = True
            status_label = "FALL_ALERT"
            if is_pose_valid and not is_horizontal:
                # Subject recovered / stood up
                tracker.state = "NORMAL"
                tracker.frames_in_horizontal = 0
                tracker.alert_latched = False

        elif tracker.state == "LYING_DOWN":
            if is_pose_valid and not is_horizontal:
                tracker.state = "NORMAL"
                tracker.frames_in_horizontal = 0
                tracker.alert_latched = False
            elif has_recent_descent:
                tracker.state = "RAPID_DESCENT"
                tracker.frames_in_horizontal += 1

        telemetry = {
            "velocity": downward_velocity,
            "torso_angle": torso_angle if torso_angle is not None else 0.0,
            "aspect_ratio": aspect_ratio
        }
        return alert_triggered, status_label, telemetry

    def process_frame(self, frame: np.ndarray, timestamp_str: str = ""):
        """
        Runs tracking, extracts pose, evaluates fall heuristics, and draws annotations.
        Returns: (annotated_frame, new_fall_events)
        """
        self.frame_count += 1
        new_fall_events = []

        # Safely remove tracks that have disappeared (>60 frames / ~2s without detection)
        self.cleanup_lost_tracks(max_lost_frames=60)

        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=self.device
            )
        except Exception:
            return frame.copy(), []

        annotated_frame = frame.copy()

        if (
            results
            and len(results) > 0
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else [0.8] * len(track_ids)

            for i, track_id in enumerate(track_ids):
                if track_id not in self.trackers:
                    self.trackers[track_id] = PersonTrackHistory(track_id)

                tracker = self.trackers[track_id]
                kpts = keypoints[i]
                box = boxes[i]
                det_conf = float(confs[i]) if i < len(confs) else 0.8

                center_y, b_w, b_h, ar, angle, is_valid = self.extract_features(kpts, box)
                tracker.update(self.frame_count, center_y, b_w, b_h, angle, is_valid)

                is_alert, status_label, telemetry = self._evaluate_tracker(tracker)

                # Only trigger a new fall event ONCE per incident until reset/recovered
                if is_alert and not tracker.alert_latched:
                    tracker.alert_latched = True
                    tracker.fall_timestamp = timestamp_str
                    new_fall_events.append({
                        "track_id": track_id,
                        "timestamp": timestamp_str,
                        "torso_angle": telemetry.get("torso_angle", 0.0),
                        "descent_velocity": telemetry.get("velocity", 0.0),
                        "aspect_ratio": telemetry.get("aspect_ratio", 0.0),
                        "confidence": det_conf
                    })

                # Visual color codes: Red (Fall Alert), Yellow (Rapid Descent), Orange (Lying Down), Green (Normal)
                if tracker.state == "FALL_ALERT":
                    color = (0, 0, 255)
                    tag = "FALL DETECTED"
                elif tracker.state == "RAPID_DESCENT":
                    color = (0, 200, 255)
                    tag = "RAPID DESCENT"
                elif tracker.state == "LYING_DOWN":
                    color = (255, 165, 0)
                    tag = "PRONE / RESTING"
                else:
                    color = (0, 255, 0)
                    tag = "NORMAL"

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                v_disp = telemetry.get("velocity", 0.0)
                ang_disp = telemetry.get("torso_angle", 0.0)
                label_text = f"ID:{track_id} | {tag} | v:{v_disp:+.1f} | ang:{ang_disp:.0f}deg"
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2
                )

                # Draw skeleton links
                for p1, p2 in SKELETON_PAIRS:
                    if kpts[p1][2] > 0.4 and kpts[p2][2] > 0.4:
                        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                        cv2.line(annotated_frame, pt1, pt2, (255, 180, 0), 2)

                # Draw keypoints
                for pt in kpts:
                    if pt[2] > 0.4:
                        cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        return annotated_frame, new_fall_events

    def cleanup_lost_tracks(self, max_lost_frames: int = 60):
        """Safely removes trackers that have not been observed for max_lost_frames."""
        stale_ids = [
            tid for tid, tracker in self.trackers.items()
            if (self.frame_count - tracker.last_seen_frame) > max_lost_frames
        ]
        for tid in stale_ids:
            del self.trackers[tid]
        return stale_ids

    def acknowledge_tracker_alerts(self):
        """Acknowledges alerts: keeps latches so the same ongoing fall cannot
        re-trigger.  Latches clear automatically when the person recovers
        to an upright posture (any recovery-to-NORMAL transition)."""
        for tracker in self.trackers.values():
            if tracker.state == "FALL_ALERT":
                tracker.state = "LYING_DOWN"
            tracker.frames_in_horizontal = 0
            tracker.recent_descents.clear()
        self.recent_descents.clear()

    def reset_tracker_alerts(self):
        """Fully resets all tracker alert and posture state for a clean restart."""
        for tracker in self.trackers.values():
            tracker.alert_latched = False
            tracker.state = "NORMAL"
            tracker.frames_in_horizontal = 0
            tracker.recent_descents.clear()
        self.recent_descents.clear()

    def clear_all_tracks(self):
        """Completely clears all tracker histories, frame counts, and state for fresh test."""
        self.trackers.clear()
        self.frame_count = 0
        self.recent_descents.clear()
