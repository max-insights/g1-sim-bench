"""TEST ONLY (tests/lat_sim_test.sh): fake XRoboToolkit SDK for the latency-monitor sim test (no headset). Controllers move sinusoidally;
A+X is held 4.0-4.8 s after init (arm following on). PICO samples at 72 Hz, delivered 20-50 ms late (simulated Wi-Fi),
timestamps on the wall clock (as if the headset clock were synchronised)."""
import math
import os
import random
import time

_t0 = None
HZ = 72.0


def _now():
    return time.time()


def init():
    global _t0
    _t0 = _now()


def close():
    pass


_deliv = {}


def _latest_sample():
    """index and creation time of the newest sample that has already 'arrived'"""
    t = _now()
    k = int((t - _t0) * HZ)
    while k > 0:
        if k not in _deliv:
            _deliv[k] = random.uniform(0.020, 0.050)
        s = _t0 + k / HZ
        if s + _deliv[k] <= t:
            return k, s
        k -= 1
    return 0, _t0


def _el():
    return _latest_sample()[1] - _t0


# FAKE_PICO_PROFILE=fast (2026-11-24, tests/speed_ab.sh): quick reaches instead of the slow sine, to drive the arm
# joints into the --upper-body-joint-speed cap like fast real teleop. Every 2.5 s a min-jerk move to one of these
# offsets (m; x out, y up, z back) and 1.0 s later back; peak hand speed 1.9 * distance / duration = 0.9 - 1.9 m/s.
_REACH = [((0.00, 0.00, -0.30), 0.35), ((0.00, 0.25, 0.00), 0.30), ((0.25, 0.00, -0.10), 0.35),
          ((0.15, 0.20, -0.25), 0.30), ((0.00, -0.15, -0.20), 0.40), ((0.30, 0.10, 0.00), 0.30)]


def _minjerk(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def _fast(e, side):
    if e < 6.0:
        return 0.0, 0.0, 0.0
    k, r = divmod(e - 6.0, 2.5)
    (dx, dy, dz), dur = _REACH[(int(k) + (0 if side == "r" else 3)) % len(_REACH)]
    w = _minjerk(r / dur) - _minjerk((r - 1.0) / dur)
    return (dx if side == "r" else -dx) * w, dy * w, dz * w


def _pose(side):
    e = _el()
    if os.environ.get("FAKE_PICO_PROFILE") == "fast":
        dx, dy, dz = _fast(e, side)
        return [(-0.20 if side == "l" else 0.20) + dx, 1.0 + dy, -0.30 + dz, 0.0, 0.0, 0.0, 1.0]
    a = 0.0 if e < 6.0 else 0.10 * math.sin(2 * math.pi * 0.5 * (e - 6.0))
    x = (-0.20 if side == "l" else 0.20) + (a if side == "r" else -a) * 0.5
    return [x, 1.0 + a, -0.30, 0.0, 0.0, 0.0, 1.0]


def get_left_controller_pose():
    return _pose("l")


def get_right_controller_pose():
    return _pose("r")


def get_headset_pose():
    return [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0]


def _ax():
    return 4.0 <= _el() <= 4.8


get_A_button = get_X_button = _ax
get_B_button = get_Y_button = lambda: False
get_left_menu_button = get_right_menu_button = lambda: False
get_left_axis_click = get_right_axis_click = lambda: False
get_left_trigger = get_right_trigger = get_left_grip = get_right_grip = lambda: 0.0
get_left_axis = get_right_axis = lambda: [0.0, 0.0]
get_left_hand_is_active = get_right_hand_is_active = lambda: False
get_left_hand_tracking_state = get_right_hand_tracking_state = lambda: None
num_motion_data_available = lambda: 0
is_body_data_available = lambda: False


def get_time_stamp_ns():
    return int(_latest_sample()[1] * 1e9)
