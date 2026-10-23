"""
Focused unit tests for multi-person fall detection isolation and track lifecycle in fall_detector.py.
Verifies that:
  1. Descent history is tracked individually per person; one person's rapid descent cannot affect another person.
  2. A person resting or prone does NOT get falsely triggered into RAPID_DESCENT or FALL_ALERT when another person jumps or falls.
  3. One person falling triggers FALL_ALERT for only that person; other people in frame remain NORMAL.
  4. Track history and descent buffers are safely cleaned up when tracks disappear.
"""

import pytest
from unittest.mock import patch, MagicMock
from fall_detector import RealtimeFallDetector, PersonTrackHistory


@pytest.fixture
def detector():
    """RealtimeFallDetector with YOLO mocked out."""
    with patch("fall_detector.YOLO") as MockYOLO:
        MockYOLO.return_value = MagicMock()
        det = RealtimeFallDetector(model_path="dummy.pt")
    return det


def _push(tracker, detector, n, center_y, w, h, angle, is_valid=True):
    """Pushes n frame updates to a tracker."""
    for _ in range(n):
        detector.frame_count += 1
        tracker.update(detector.frame_count, center_y, w, h, angle, is_valid)


class TestMultiPersonIsolation:

    def test_one_person_rapid_descent_does_not_affect_second_person(self, detector):
        """Person 1 drops rapidly. Person 2 is already horizontal / resting.
        Person 2 must NOT be dragged into RAPID_DESCENT or FALL_ALERT by Person 1's movement."""
        p1 = PersonTrackHistory(track_id=1)
        p2 = PersonTrackHistory(track_id=2)
        detector.trackers[1] = p1
        detector.trackers[2] = p2

        # Initialize both trackers with baseline frames
        for _ in range(6):
            detector.frame_count += 1
            # P1: standing upright
            p1.update(detector.frame_count, center_y=100.0, box_w=50, box_h=150, torso_angle=5.0, is_valid=True)
            # P2: resting prone on sofa/bed (horizontal, but no vertical motion)
            p2.update(detector.frame_count, center_y=300.0, box_w=180, box_h=70, torso_angle=75.0, is_valid=True)

        detector._evaluate_tracker(p1)
        detector._evaluate_tracker(p2)
        assert p1.state == "NORMAL"
        assert p2.state == "LYING_DOWN"

        # P1 now performs a rapid descent (center_y jumps from 100 to 200, velocity = +100)
        detector.frame_count += 1
        p1.update(detector.frame_count, center_y=200.0, box_w=50, box_h=150, torso_angle=5.0, is_valid=True)
        # P2 remains stationary (center_y stays at 300, velocity = 0)
        p2.update(detector.frame_count, center_y=300.0, box_w=180, box_h=70, torso_angle=75.0, is_valid=True)

        alert1, state1, _ = detector._evaluate_tracker(p1)
        alert2, state2, _ = detector._evaluate_tracker(p2)

        # Person 1 must detect rapid descent
        assert p1.state == "RAPID_DESCENT"
        assert len(p1.recent_descents) == 1

        # Person 2 must NOT have recent descents and must NOT transition to RAPID_DESCENT
        assert len(p2.recent_descents) == 0, "Person 2 must not inherit Person 1's descent history"
        assert p2.state == "LYING_DOWN", f"Person 2 should remain LYING_DOWN, but got {p2.state}"
        assert alert2 is False

    def test_person1_fall_does_not_trigger_alert_for_person2(self, detector):
        """Person 1 experiences a full fall sequence while Person 2 walks normally.
        Only Person 1 triggers an alert; Person 2 remains NORMAL throughout."""
        p1 = PersonTrackHistory(track_id=1)
        p2 = PersonTrackHistory(track_id=2)
        detector.trackers[1] = p1
        detector.trackers[2] = p2

        # 6 baseline frames
        for _ in range(6):
            detector.frame_count += 1
            p1.update(detector.frame_count, center_y=100.0, box_w=50, box_h=150, torso_angle=5.0, is_valid=True)
            p2.update(detector.frame_count, center_y=120.0, box_w=50, box_h=150, torso_angle=8.0, is_valid=True)

        # P1 rapid downward jump; P2 steps forward normally
        detector.frame_count += 1
        p1.update(detector.frame_count, center_y=200.0, box_w=50, box_h=150, torso_angle=10.0, is_valid=True)
        p2.update(detector.frame_count, center_y=122.0, box_w=50, box_h=150, torso_angle=8.0, is_valid=True)

        detector._evaluate_tracker(p1)
        detector._evaluate_tracker(p2)
        assert p1.state == "RAPID_DESCENT"
        assert p2.state == "NORMAL"

        # P1 remains horizontal on ground for resting_frames_req (3 frames)
        p1_alerts = []
        p2_alerts = []
        for _ in range(detector.resting_frames_req):
            detector.frame_count += 1
            p1.update(detector.frame_count, center_y=210.0, box_w=180, box_h=70, torso_angle=75.0, is_valid=True)
            p2.update(detector.frame_count, center_y=124.0, box_w=50, box_h=150, torso_angle=8.0, is_valid=True)

            a1, _, _ = detector._evaluate_tracker(p1)
            a2, _, _ = detector._evaluate_tracker(p2)
            p1_alerts.append(a1)
            p2_alerts.append(a2)

        # Person 1 triggers fall alert
        assert any(p1_alerts) is True
        assert p1.state == "FALL_ALERT"

        # Person 2 never triggers alert and remains NORMAL
        assert any(p2_alerts) is False
        assert p2.state == "NORMAL"


class TestTrackCleanup:

    def test_cleanup_lost_tracks_removes_stale_trackers(self, detector):
        """Trackers not observed for > max_lost_frames must be purged safely from memory."""
        p1 = PersonTrackHistory(track_id=1)
        p2 = PersonTrackHistory(track_id=2)
        detector.trackers[1] = p1
        detector.trackers[2] = p2

        # Both present at frame 10
        detector.frame_count = 10
        p1.update(10, 100, 50, 150, 5.0, True)
        p2.update(10, 100, 50, 150, 5.0, True)

        # Now simulate time passing up to frame 75:
        # P1 is continuously updated; P2 disappeared (last seen at frame 10)
        detector.frame_count = 75
        p1.update(75, 102, 50, 150, 5.0, True)

        # Cleanup with threshold = 60 frames (75 - 10 = 65 > 60)
        stale = detector.cleanup_lost_tracks(max_lost_frames=60)

        assert stale == [2], "Person 2 should be identified as stale and removed"
        assert 2 not in detector.trackers, "Person 2 tracker should be purged"
        assert 1 in detector.trackers, "Person 1 tracker must remain active"

    def test_clear_all_tracks_resets_everything(self, detector):
        """clear_all_tracks() must clear trackers dictionary, frame count, and descents."""
        p1 = PersonTrackHistory(track_id=1)
        p1.recent_descents.append({"frame": 5, "y": 100.0})
        detector.trackers[1] = p1
        detector.frame_count = 50
        detector.recent_descents.append({"frame": 5, "y": 100.0})

        detector.clear_all_tracks()

        assert len(detector.trackers) == 0
        assert detector.frame_count == 0
        assert len(detector.recent_descents) == 0
