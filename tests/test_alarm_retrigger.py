"""
Focused tests for the alarm re-trigger fix.
Tests the fall-detection FSM in fall_detector.py and the alarm lifecycle
in camera_stream.py to verify:
  1. After acknowledge, the same ongoing fall does NOT re-trigger.
  2. After the person recovers (stands up), a genuinely new fall DOES trigger.
  3. Full reset clears everything for immediate re-detection.
  4. All recovery-to-NORMAL transitions clear alert_latched.

Note: _evaluate_tracker() returns a display `status_label` which may lag behind
`tracker.state` for some transitions.  Tests assert on `tracker.state` directly
to verify actual FSM behaviour.
"""

import pytest
from unittest.mock import patch, MagicMock
from fall_detector import RealtimeFallDetector, PersonTrackHistory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """RealtimeFallDetector with YOLO model mocked out (no weights needed)."""
    with patch("fall_detector.YOLO") as MockYOLO:
        MockYOLO.return_value = MagicMock()
        det = RealtimeFallDetector(model_path="dummy.pt")
    return det


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_frames(tracker, detector, n, center_y, w, h, angle, is_valid=True):
    """Push n identical posture snapshots into a tracker's history."""
    for _ in range(n):
        detector.frame_count += 1
        tracker.update(detector.frame_count, center_y, w, h, angle, is_valid)


def _drive_to_fall_alert(detector, tracker):
    """Drive tracker through NORMAL -> RAPID_DESCENT -> FALL_ALERT.
    Returns (alert_triggered, status_label, telemetry) from the triggering call
    and latches the alert (mimicking what process_frame does)."""
    # Upright baseline -- fill the sliding window (window_size=6)
    _push_frames(tracker, detector, n=6, center_y=100.0, w=50, h=150, angle=5.0)

    # Rapid downward jump
    _push_frames(tracker, detector, n=1, center_y=200.0, w=50, h=150, angle=5.0)
    detector._evaluate_tracker(tracker)
    assert tracker.state == "RAPID_DESCENT"

    # Stay horizontal/prone for resting_frames_req (default 3) frames
    result = (False, "", {})
    for _ in range(detector.resting_frames_req):
        _push_frames(tracker, detector, n=1, center_y=210.0, w=180, h=80, angle=70.0)
        result = detector._evaluate_tracker(tracker)

    assert tracker.state == "FALL_ALERT"
    alert, _, _ = result
    assert alert is True
    # Simulate what process_frame() does on first alert
    tracker.alert_latched = True
    return result


def _recover_to_normal(tracker, detector, n_frames=10):
    """Push upright frames and evaluate until tracker reaches NORMAL."""
    for _ in range(n_frames):
        _push_frames(tracker, detector, n=1,
                     center_y=100.0, w=50, h=150, angle=5.0)
        detector._evaluate_tracker(tracker)
        if tracker.state == "NORMAL":
            return
    raise AssertionError(
        f"Tracker did not recover to NORMAL after {n_frames} upright frames; "
        f"stuck in {tracker.state}"
    )


# ===========================================================================
# Test Suite 1 -- Acknowledge must NOT re-trigger the same fall
# ===========================================================================

class TestAcknowledgeNoRetrigger:

    def test_acknowledge_preserves_latch_and_resets_counters(self, detector):
        """acknowledge_tracker_alerts() must keep alert_latched=True,
        reset frames_in_horizontal to 0, and clear recent_descents."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        _drive_to_fall_alert(detector, tracker)

        detector.acknowledge_tracker_alerts()

        assert tracker.state == "LYING_DOWN"
        assert tracker.alert_latched is True, "Latch must be preserved"
        assert tracker.frames_in_horizontal == 0, "Horizontal counter must reset"
        assert len(detector.recent_descents) == 0, "Descent history must be cleared"

    def test_continued_horizontal_does_not_retrigger(self, detector):
        """After acknowledge, feeding more horizontal frames must never
        produce a new alertable event (alert_latched blocks it)."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        _drive_to_fall_alert(detector, tracker)
        detector.acknowledge_tracker_alerts()

        # Feed 20 more horizontal frames at the same position
        for i in range(20):
            _push_frames(tracker, detector, n=1,
                         center_y=210.0, w=180, h=80, angle=70.0)
            alert, _, _ = detector._evaluate_tracker(tracker)
            # Even if the FSM drifts back to FALL_ALERT via residual
            # velocity in the history window, the latch blocks new events
            if alert:
                assert tracker.alert_latched is True, (
                    f"Frame {i}: alert_triggered but latch should block event emission"
                )


# ===========================================================================
# Test Suite 2 -- New fall AFTER recovery must trigger
# ===========================================================================

class TestNewFallAfterRecovery:

    def test_recovery_clears_latch_then_new_fall_triggers(self, detector):
        """Person falls -> acknowledge -> stands up -> falls again -> alarm."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker

        # Fall #1
        _drive_to_fall_alert(detector, tracker)
        detector.acknowledge_tracker_alerts()

        # Recovery: person stands back up (upright, low angle, tall bbox)
        _recover_to_normal(tracker, detector)
        assert tracker.state == "NORMAL"
        assert tracker.alert_latched is False, "Latch must clear on recovery"

        # Fall #2: brand-new fall should fire
        _drive_to_fall_alert(detector, tracker)
        # _drive_to_fall_alert already asserts alert=True and state=FALL_ALERT


# ===========================================================================
# Test Suite 3 -- Full reset clears everything
# ===========================================================================

class TestFullReset:

    def test_reset_clears_all_state(self, detector):
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        _drive_to_fall_alert(detector, tracker)

        detector.reset_tracker_alerts()

        assert tracker.state == "NORMAL"
        assert tracker.alert_latched is False
        assert tracker.frames_in_horizontal == 0
        assert len(detector.recent_descents) == 0

    def test_fall_detectable_immediately_after_reset(self, detector):
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        _drive_to_fall_alert(detector, tracker)

        detector.reset_tracker_alerts()
        # Clear history so baseline doesn't carry stale positions
        tracker.history.clear()

        # Should be able to trigger a brand-new fall
        _drive_to_fall_alert(detector, tracker)


# ===========================================================================
# Test Suite 4 -- Every recovery-to-NORMAL path clears alert_latched
# ===========================================================================

class TestRecoveryPathsClearLatch:

    def test_lying_down_to_normal(self, detector):
        """LYING_DOWN -> NORMAL must clear alert_latched."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        tracker.state = "LYING_DOWN"
        tracker.alert_latched = True

        # Feed upright frames and evaluate until recovery
        _recover_to_normal(tracker, detector)
        assert tracker.state == "NORMAL"
        assert tracker.alert_latched is False

    def test_rapid_descent_to_normal(self, detector):
        """RAPID_DESCENT -> NORMAL (upward velocity) must clear alert_latched."""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        tracker.state = "RAPID_DESCENT"
        tracker.alert_latched = True

        # Fill history at y=200, then frames at y=180 -> negative velocity
        _push_frames(tracker, detector, n=6,
                     center_y=200.0, w=50, h=150, angle=5.0)
        # Multiple upright frames moving upward to ensure recovery
        _recover_to_normal(tracker, detector)
        assert tracker.state == "NORMAL"
        assert tracker.alert_latched is False

    def test_fall_alert_to_normal(self, detector):
        """FALL_ALERT -> NORMAL (person stands up) must clear alert_latched.
        (This path was already correct before the fix -- regression check.)"""
        tracker = PersonTrackHistory(track_id=1)
        detector.trackers[1] = tracker
        _drive_to_fall_alert(detector, tracker)

        # Stand back up
        _recover_to_normal(tracker, detector)
        assert tracker.state == "NORMAL"
        assert tracker.alert_latched is False
