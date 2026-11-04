"""Deterministic marker skill. The agent picks a mode; this picks the velocity."""

MODES = ('report', 'preview', 'follow', 'approach')
_REPORT_ALIASES = ('stop', 'idle', 'estop', 'halt', 'report')
_PREVIEW_ALIASES = ('preview', 'command')


def normalize_mode(text):
    """Return a mode name, or None when the string is not a command."""
    mode = (text or '').strip().lower()
    if mode in _REPORT_ALIASES:
        return 'report'
    if mode in _PREVIEW_ALIASES:
        return 'preview'
    if mode in ('follow', 'approach'):
        return mode
    return None


def clamp(value, limit):
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def slew(current, target, max_delta):
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


def compute_command(
    mode,
    found,
    bearing,
    distance_m,
    *,
    latched,
    target_distance=1.0,
    linear_gain=0.45,
    angular_gain=1.2,
    max_linear=0.25,
    max_angular=0.6,
    bearing_deadband=0.06,
    distance_deadband=0.05,
    approach_bearing_tol=0.10,
    approach_distance_tol=0.08,
):
    """Return (linear_x, angular_z, latched).

    Positive bearing means the marker is right of center. Positive angular.z
    is a left turn, so the yaw command is the opposite of bearing.
    Positive distance error means the marker is too far, so the robot drives forward.
    """
    if mode != 'approach':
        latched = False
    if mode not in ('follow', 'approach'):
        return 0.0, 0.0, False
    if latched:
        return 0.0, 0.0, True
    if not found:
        return 0.0, 0.0, False

    angular = 0.0
    if bearing is not None and abs(bearing) >= bearing_deadband:
        angular = clamp(-angular_gain * bearing, max_angular)

    if distance_m is None:
        return 0.0, angular, False

    distance_error = distance_m - target_distance
    linear = 0.0
    if abs(distance_error) >= distance_deadband:
        linear = clamp(linear_gain * distance_error, max_linear)

    arrived = (
        mode == 'approach'
        and abs(bearing or 0.0) <= approach_bearing_tol
        and abs(distance_error) <= approach_distance_tol
    )
    if arrived:
        return 0.0, 0.0, True
    return linear, angular, False


def approach_phase(found, latched, armed=True, resume_in_sec=None):
    """Audience / agent label for the Act 4 loop."""
    if latched:
        return 'stopped_at_1m'
    if not armed:
        return 'cooldown'
    if not found:
        return 'waiting_for_marker'
    return 'approaching'


def approach_should_resume(latch_age_sec, resume_after_sec=5.0):
    """True when a latched approach should clear and hunt again."""
    if latch_age_sec is None:
        return False
    return latch_age_sec >= resume_after_sec
