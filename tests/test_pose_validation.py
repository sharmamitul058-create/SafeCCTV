"""
Focused unit tests for missing/low-confidence keypoints and normal activities in fall_detector.py.
Verifies that:
  1. Missing or low-confidence keypoints do NOT trigger a fall based only on bounding-box aspect ratio.
  2. Normal activities (upright sitting with wide bounding box, bending/squatting, walking) do not cause false fall alerts.
  3. Legitimate falls with valid pose data are preserved and trigger alerts correctly.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fall_detector import RealtimeFallDetector, PersonTrackHistory


@pytest.fixture
def detector():
    """RealtimeFallDetector with YOLO model mocked out."""
    with patch("fall_detector.YOLO") as MockYOLO:
        MockYOLO.return_value = MagicMock()
        det = RealtimeFallDetector(model_path="dummy.pt")
    return det


def _make_coco_keypoints(l_sh=(100, 100, 0.9), r_sh=(120, 100, 0.9),
                         l_hip=(100, 160, 0.9), r_hip=(120, 160, 0.9)):
    """Create a 17-keypoint COCO array with specified shoulder and hip coordinates."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = l_sh   # Left Shoulder
    kpts[6] = r_sh   # Right Shoulder
    kpts[11] = l_hip # Left Hip
    kpts[12] = r_hip # Right Hip
    return kpts


def _push_frames(tracker, detector, n, center_y, w, h, angle, is_valid=True):
    """Helper to push snapshots into tracker history."""
    for _ in range(n):
        detector.frame_count += 1
        tracker.update(detector.frame_count, center_y, w, h, angle, is_valid)


# ===========================================================================
# 1. Feature Extraction & Keypoint Validation Tests
# ===========================================================================

class TestKeypointValidation:

    def test_missing_keypoints_returns_invalid(self, detector):
        """None or short keypoint array returns is_valid=False and angle=None."""
        box = [100, 100, 200, 180]
        # None
        cy, w, h, ar, angle, is_valid = detector.extract_features(None, box)
        assert is_valid is False
        assert angle is None

        # Empty / truncated array
        cy, w, h, ar, angle, is_valid = detector.extract_features(np.zeros((5, 3)), box)
        assert is_valid is False
        assert angle is None

    def test_low_confidence_keypoints_returns_invalid(self, detector):
        """Keypoints with confidence below min_conf (0.4) return is_valid=False."""
        box = [100, 100, 200, 180]
        # Shoulders and hips below 0.4
        kpts = _make_coco_keypoints(
            l_sh=(100, 100, 0.2), r_sh=(120, 100, 0.1),
            l_hip=(100, 160, 0.3), r_hip=(120, 160, 0.2)
        )
        cy, w, h, ar, angle, is_valid = detector.extract_features(kpts, box, min_conf=0.4)
        assert is_valid is False
        assert angle is None

    def test_partial_confidence_uses_visible_keypoints(self, detector):
        """When one shoulder is occluded, visible shoulder is used without error."""
        box = [100, 100, 200, 250]
        # Left shoulder visible (0.85), right shoulder occluded (0.05)
        kpts = _make_coco_keypoints(
            l_sh=(100, 100, 0.85), r_sh=(0, 0, 0.05),
            l_hip=(100, 200, 0.85), r_hip=(120, 200, 0.85)
        )
        cy, w, h, ar, angle, is_valid = detector.extract_features(kpts, box, min_conf=0.4)
        assert is_valid is True
        assert angle is not None
        # Upright posture: dx is small relative to dy
        assert angle < 30.0


# ===========================================================================
# 2. Preventing False Falls on Invalid Pose & Wide Aspect Ratio
# ===========================================================================

class TestNoFallOnInvalidPose:

    def test_wide_box_with_invalid_pose_does_not_trigger_fall(self, detector):
        """Wide bounding box (aspect_ratio >= 1.05) with invalid pose must NOT trigger fall."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        # Fill baseline: upright
        _push_frames(tracker, detector, n=6, center_y=100.0, w=50, h=150, angle=5.0, is_valid=True)

        # Rapid descent with keypoints lost/occluded (e.g. crouching behind furniture)
        _push_frames(tracker, detector, n=1, center_y=200.0, w=180, h=80, angle=None, is_valid=False)
        detector._evaluate_tracker(tracker)

        # 5 consecutive frames staying wide (aspect_ratio = 180/80 = 2.25) but pose is invalid
        alerts = []
        for _ in range(5):
            _push_frames(tracker, detector, n=1, center_y=200.0, w=180, h=80, angle=None, is_valid=False)
            alert, _, _ = detector._evaluate_tracker(tracker)
            alerts.append(alert)

        assert not any(alerts), "Fall must NOT trigger when pose data is invalid"
        assert tracker.state != "FALL_ALERT"

    def test_invalid_pose_does_not_enter_lying_down_from_normal(self, detector):
        """Wide bounding box alone without valid pose must NOT set state to LYING_DOWN."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        # 10 frames with wide bounding box and invalid pose (no vertical descent)
        for _ in range(10):
            _push_frames(tracker, detector, n=1, center_y=150.0, w=150, h=80, angle=None, is_valid=False)
            detector._evaluate_tracker(tracker)

        assert tracker.state == "NORMAL", f"Expected NORMAL, but got {tracker.state}"


# ===========================================================================
# 3. Normal Activities Test Suite (No False Positives)
# ===========================================================================

class TestNormalActivities:

    def test_upright_sitting_with_wide_box_does_not_trigger_fall(self, detector):
        """Sitting in an office chair gives wide bbox (aspect_ratio >= 1.05) but upright torso (angle ~10-25 deg)."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        # Sitting down motion (modest velocity), ending in wide bbox with angle=15 deg
        _push_frames(tracker, detector, n=6, center_y=100.0, w=50, h=150, angle=5.0, is_valid=True)
        # Moderate drop into chair
        _push_frames(tracker, detector, n=1, center_y=135.0, w=120, h=90, angle=15.0, is_valid=True)
        detector._evaluate_tracker(tracker)

        # Seated for 10 frames: wide aspect ratio (120/90 = 1.33) but angle=15 deg
        for _ in range(10):
            _push_frames(tracker, detector, n=1, center_y=135.0, w=120, h=90, angle=15.0, is_valid=True)
            alert, state, _ = detector._evaluate_tracker(tracker)
            assert alert is False
            assert tracker.state != "FALL_ALERT"

    def test_walking_pacing_does_not_trigger_fall(self, detector):
        """Walking across frame with natural vertical bobbing does not trigger fall."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        for i in range(20):
            bob = 100.0 + 3.0 * (i % 2)
            _push_frames(tracker, detector, n=1, center_y=bob, w=60, h=160, angle=8.0, is_valid=True)
            alert, _, _ = detector._evaluate_tracker(tracker)
            assert alert is False
            assert tracker.state == "NORMAL"


# ===========================================================================
# 4. Valid Fall Detection Preservation
# ===========================================================================

class TestPreserveValidFallDetection:

    def test_legitimate_fall_with_valid_keypoints_triggers_alert(self, detector):
        """Rapid descent + confirmed horizontal torso with valid pose must trigger FALL_ALERT."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        # Upright standing
        _push_frames(tracker, detector, n=6, center_y=100.0, w=50, h=150, angle=5.0, is_valid=True)

        # Rapid descent
        _push_frames(tracker, detector, n=1, center_y=200.0, w=60, h=130, angle=20.0, is_valid=True)
        detector._evaluate_tracker(tracker)
        assert tracker.state == "RAPID_DESCENT"

        # Horizontal posture on ground: angle=72 deg, valid pose for resting_frames_req
        alerts = []
        for _ in range(detector.resting_frames_req):
            _push_frames(tracker, detector, n=1, center_y=210.0, w=180, h=70, angle=72.0, is_valid=True)
            alert, _, _ = detector._evaluate_tracker(tracker)
            alerts.append(alert)

        assert any(alerts), "Legitimate fall must trigger alert"
        assert tracker.state == "FALL_ALERT"
