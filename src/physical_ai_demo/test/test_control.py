"""Control-law checks. Run from the package root or via pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_ai_demo.control import (
    approach_phase,
    approach_should_resume,
    compute_command,
    normalize_mode,
    slew,
)


def test_mode_aliases():
    assert normalize_mode(' Follow ') == 'follow'
    assert normalize_mode('STOP') == 'report'
    assert normalize_mode('estop') == 'report'
    assert normalize_mode('command') == 'preview'
    assert normalize_mode('dance') is None


def test_preview_computes_nothing_for_the_motors():
    linear, angular, latched = compute_command('preview', True, 0.5, 2.0, latched=False)
    assert (linear, angular, latched) == (0.0, 0.0, False)


def test_report_holds_still():
    linear, angular, latched = compute_command('report', True, 0.4, 2.0, latched=True)
    assert (linear, angular, latched) == (0.0, 0.0, False)


def test_follow_turns_toward_a_marker_on_the_right_and_drives_closer():
    linear, angular, latched = compute_command('follow', True, 0.5, 2.0, latched=False)
    assert linear > 0.0
    assert angular < 0.0
    assert latched is False
    assert abs(linear) <= 0.25
    assert abs(angular) <= 0.6


def test_follow_backs_away_when_too_close():
    linear, angular, _latched = compute_command('follow', True, 0.0, 0.4, latched=False)
    assert linear < 0.0
    assert angular == 0.0


def test_follow_holds_near_one_meter():
    linear, angular, latched = compute_command('follow', True, 0.0, 1.0, latched=False)
    assert linear == 0.0
    assert angular == 0.0
    assert latched is False


def test_lost_marker_stops_without_latching():
    linear, angular, latched = compute_command('approach', False, 0.4, 2.0, latched=False)
    assert (linear, angular, latched) == (0.0, 0.0, False)


def test_approach_latches_at_the_setpoint():
    linear, angular, latched = compute_command('approach', True, 0.02, 1.02, latched=False)
    assert (linear, angular, latched) == (0.0, 0.0, True)


def test_latched_approach_ignores_later_motion():
    linear, angular, latched = compute_command('approach', True, 0.8, 2.5, latched=True)
    assert (linear, angular, latched) == (0.0, 0.0, True)


def test_slew_limits_a_step():
    assert slew(0.0, 1.0, 0.2) == 0.2
    assert slew(0.2, 0.0, 0.2) == 0.0


def test_approach_resume_timer():
    assert approach_should_resume(None) is False
    assert approach_should_resume(4.9, 5.0) is False
    assert approach_should_resume(5.0, 5.0) is True


def test_approach_phase_labels():
    assert approach_phase(False, False) == 'waiting_for_marker'
    assert approach_phase(True, False) == 'approaching'
    assert approach_phase(True, True) == 'stopped_at_1m'
    assert approach_phase(True, False, armed=False) == 'cooldown'
