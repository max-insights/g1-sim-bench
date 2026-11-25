"""Single-step mode for the decoupled-WBC chain (2026-11-23). Installed by dwbc_run.py when DWBC_STEP=1.

Operator (PICO):
  left stick CLICK          toggles single-step mode on / off (printed + spoken on the workstation)
  in single-step mode       a left-stick flick (past 60 %, then back under 25 % to re-arm) = ONE step in that
                            direction: forward / back / left / right.
                            right stick left / right: a FLICK (released within turn.hold_s) = turn turn.step_deg
                            (15 deg), closed loop on the IMU yaw, flicks add up; HELD = continuous turning, faster the
                            further it is pushed (up to turn.hold_speed); release = stop at the next lift-off.
                            Continuous walking from the sticks is off.
Rise speed (2026-11-24): while Y raises the base height, the rise is limited to rise.speed (and rise.slow_speed in the
last rise.slow_zone below standing). Settings: step_profiles.json "turn" / "rise" (Mac app 设置 → 单步移动).
Keyboard (control loop, e.g. `dwbc_real.sh key t`):  t = forward, g = back, f = left, h = right, r / y = turn left /
right.

One "step" = each foot lifts once and the feet end about side by side (dwbc_step.PROFILES, tuned in MuJoCo:
forward 6-10 cm, back 5-10 cm, left ~8 cm, right ~10 cm). The control loop closes the loop on the feet (forward
kinematics of the measured joints), so it does not depend on stick timing or the PICO -> robot latency.

teleop side: PicoStreamer output gets control_data["dwbc_step"] = {"seq": n, "dir": d} (seq grows by one per flick)
             and navigate_cmd = 0 while the mode is on.
control side: G1DecoupledWholeBodyPolicy.set_goal picks up new seqs, G1GearWbcPolicy.get_action overrides the
             navigate command while a step runs (also in keyboard mode).
"""
import os
import subprocess
import time

import numpy as np

from dwbc_step import DIRS, EXTRAS, PROFILE_FILE, SingleStep, foot_pose, load_extras, load_profiles, yaw_from_quat_wxyz

KEYS = {"t": "fwd", "g": "back", "f": "left", "h": "right", "r": "turn_left", "y": "turn_right"}
_mono = time.monotonic
FLICK_ON, FLICK_OFF = 0.6, 0.25


def _say(text):
    if os.environ.get("DWBC_QUIET") == "1":
        return
    try:
        subprocess.Popen(["spd-say", "-l", "zh", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------- control loop side
def install_control():
    from decoupled_wbc.control.policy import g1_decoupled_whole_body_policy as _DW
    from decoupled_wbc.control.policy import g1_gear_wbc_policy as _GW
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    fk_model = instantiate_g1_robot_model(waist_location="lower_body")   # own instance: never touch the policy's FK cache
    profiles, src, errs = load_profiles()
    extras0, _s, _e = load_extras()
    cache = {"mtime": None, "profiles": profiles, "extras": extras0}

    def current_profiles():
        """re-read step_profiles.json (Mac app 设置 → 单步移动) when it changed; a bad file keeps the defaults"""
        try:
            mt = os.stat(PROFILE_FILE).st_mtime
        except OSError:
            mt = None
        if mt != cache["mtime"]:
            cache["mtime"] = mt
            p, src, errs = load_profiles()
            for e in errs:
                print(f"[dwbc_step] step_profiles.json ignored: {e}", flush=True)
            if p != cache["profiles"]:
                print("[dwbc_step] profiles (" + src + "): "
                      + ", ".join(f"{d} {v[0]} {'rad/s' if d.startswith('turn') else 'm/s'} off@{v[1]}#{v[2]}" for d, v in p.items()), flush=True)
            cache["profiles"] = p
            x, _s, xerrs = load_extras()
            for e in xerrs:
                print(f"[dwbc_step] step_profiles.json turn / rise ignored: {e}", flush=True)
            if x["turn"] != cache["extras"]["turn"]:
                print(f"[dwbc_step] turn settings: {x['turn']}", flush=True)
            cache["extras"] = x
        return cache["profiles"]
    G = _GW.G1GearWbcPolicy
    D = _DW.G1DecoupledWholeBodyPolicy
    orig_get_action, orig_key, orig_set_goal = G.get_action, G.handle_keyboard_button, D.set_goal

    def _ss(self):
        if getattr(self, "_dwbc_step", None) is None:
            self._dwbc_step = SingleStep(profiles=profiles, turn=extras0["turn"])
        return self._dwbc_step

    def _request(self, direction, source):
        if not getattr(self, "use_policy_action", False):
            print(f"[dwbc_step] {direction} ignored ({source}): leg policy is off", flush=True)
            return
        ss = _ss(self)
        if not ss.busy:
            ss.profiles = current_profiles()
            ss.turn = dict(cache["extras"]["turn"])
        # the control loop's own clock (get_action's time argument), so sim / tests faster than real time work too
        ss.request(getattr(self, "_dwbc_now", time.monotonic()), direction)

    def get_action(self, time=None, arms_target_pose=None, base_height_command=None, torso_orientation_rpy=None,
                   interpolated_navigate_cmd=None):
        ss = getattr(self, "_dwbc_step", None)
        if ss is not None and self.observation is not None:
            now = float(time) if time is not None else _mono()
            self._dwbc_now = now
            if not self.use_policy_action:
                ss.cancel(now, "leg policy off")
            o = self.observation
            quat = o["floating_base_pose"][3:7]
            zl, zr, dx = foot_pose(fk_model, o["q"], quat)
            ov = ss.update(now, zl, zr, dx, yaw=yaw_from_quat_wxyz(quat))
            if ov is not None:
                interpolated_navigate_cmd = np.asarray(ov, dtype=float)
                self.cmd = interpolated_navigate_cmd.copy()     # keyboard mode too (teleop cmd not in use)
        return orig_get_action(self, time, arms_target_pose, base_height_command, torso_orientation_rpy,
                               interpolated_navigate_cmd)

    def handle_keyboard_button(self, key):
        if key in KEYS:
            _request(self, KEYS[key], f"key {key}")
            return
        return orig_key(self, key)

    def set_goal(self, goal):
        s = goal.get("dwbc_step") if isinstance(goal, dict) else None
        if s:
            seq = int(s.get("seq", 0))
            last = getattr(self, "_dwbc_step_seq", None)
            self._dwbc_step_seq = seq
            # first message after a (re)start only syncs the counter: never replay an old flick
            if last is not None and seq != last and s.get("dir") in DIRS:
                _request(self.lower_body_policy, s["dir"], "PICO")
            hv = float(s.get("turn_hold", 0.0) or 0.0)
            lp = self.lower_body_policy
            ss = getattr(lp, "_dwbc_step", None)
            if ss is not None and (hv != 0.0 or (ss.busy and ss._stop_on == "hold")):
                if hv != 0.0 and not getattr(lp, "use_policy_action", False):
                    pass                                   # leg policy off: never start turning
                else:
                    if hv != 0.0 and not ss.busy:
                        ss.profiles = current_profiles()
                        ss.turn = dict(cache["extras"]["turn"])
                    ss.set_hold(getattr(lp, "_dwbc_now", time.monotonic()), hv)
        return orig_set_goal(self, goal)

    G.get_action, G.handle_keyboard_button, D.set_goal = get_action, handle_keyboard_button, set_goal
    # make sure the state machine exists from the start (foot reference tracked while standing)
    orig_init = G.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._dwbc_step = SingleStep(profiles=profiles, turn=extras0["turn"])

    G.__init__ = __init__
    print(f"[dwbc_step] turn: flick = {extras0['turn']['step_deg']:.0f} deg (closed loop on the yaw), held = up to "
          f"{extras0['turn']['hold_speed']:.2f} rad/s", flush=True)
    print("[dwbc_step] control: single step on keys t/g/f/h and PICO flicks "
          + ", ".join(f"{d} {p[0]} {'rad/s' if d.startswith('turn') else 'm/s'} off@{p[1]}#{p[2]}" for d, p in profiles.items()), flush=True)


# ----------------------------------------------------------------------------------------------------- teleop side
def install_teleop():
    from decoupled_wbc.control.teleop.streamers import pico_streamer as _PS

    P = _PS.PicoStreamer
    orig_gen = P._generate_unified_raw_data
    xcache = {"mtime": -1, "x": load_extras()[0]}

    def extras():
        try:
            mt = os.stat(PROFILE_FILE).st_mtime
        except OSError:
            mt = None
        if mt != xcache["mtime"]:
            xcache["mtime"] = mt
            x, _src, errs = load_extras()
            for e in errs:
                print(f"[dwbc_step] step_profiles.json turn / rise ignored: {e}", flush=True)
            if x != xcache["x"]:
                print(f"[dwbc_step] teleop settings: turn {x['turn']}, rise {x['rise']}", flush=True)
            xcache["x"] = x
        return xcache["x"]

    def limit_rise(self, h_prev, out, now):
        """base height going up (Y held / A+B+X+Y ramp): at most rise.speed (rise.slow_speed in the slow zone)"""
        h_new = getattr(self, "current_base_height", None)
        last = getattr(self, "_dwbc_rise_t", None)
        self._dwbc_rise_t = now
        if h_prev is None or h_new is None or h_new <= h_prev + 1e-9:
            return
        r = extras()["rise"]
        dt = 0.05 if last is None else min(0.2, max(0.0, now - last))
        top = 0.74
        rate = r["slow_speed"] if (r["slow_zone"] > 0 and h_prev >= top - r["slow_zone"] - 1e-9) else r["speed"]
        h = min(float(h_new), float(h_prev) + rate * dt)
        if r["slow_zone"] > 0 and h_prev < top - r["slow_zone"] < h:      # crossing into the slow zone this frame
            h = top - r["slow_zone"]
        if h < h_new - 1e-9:
            self.current_base_height = h
            cd = out.control_data
            if "base_height_command" in cd:
                cd["base_height_command"] = h

    def gen(self, pico_data):
        h_prev = getattr(self, "current_base_height", None)
        out = orig_gen(self, pico_data)
        now = _mono()
        limit_rise(self, h_prev, out, now)
        st = getattr(self, "_dwbc_step_st", None)
        if st is None:
            st = self._dwbc_step_st = {"on": False, "click": False, "armed": True, "armed_r": True, "seq": 0, "dir": None,
                                       "r_t": None, "r_sign": 0, "r_hold": False}
        hold_v = 0.0
        click = bool(pico_data.get("left_axis_click", False))
        if click and not st["click"]:
            st["on"] = not st["on"]
            st["armed"] = True
            print(f"[dwbc_step] PICO left stick click -> single-step mode {'ON' if st['on'] else 'OFF'}", flush=True)
            _say("单步模式" if st["on"] else "连续行走")
        st["click"] = click
        if st["on"]:
            x, y = (float(v) for v in pico_data["left_joystick"][:2])
            mag = max(abs(x), abs(y))
            if st["armed"] and mag > FLICK_ON and not click:
                d = ("fwd" if y > 0 else "back") if abs(y) >= abs(x) else ("left" if x < 0 else "right")
                st["seq"] += 1
                st["dir"] = d
                st["armed"] = False
                print(f"[dwbc_step] PICO flick #{st['seq']} -> {d}", flush=True)
            elif mag < FLICK_OFF:
                st["armed"] = True
            # right stick left / right (2026-11-24): flick = turn by angle, held = continuous turning
            rx, ry = (float(v) for v in pico_data["right_joystick"][:2])
            t_cfg = extras()["turn"]
            if st["r_t"] is None:
                if st["armed_r"] and abs(rx) > FLICK_ON and abs(rx) >= abs(ry):
                    st["r_t"], st["r_sign"], st["r_hold"], st["armed_r"] = now, (1 if rx < 0 else -1), False, False
            if st["r_t"] is not None:
                if max(abs(rx), abs(ry)) < FLICK_OFF:          # released
                    if not st["r_hold"]:
                        d = "turn_left" if st["r_sign"] > 0 else "turn_right"    # streamer: yaw = -right_joystick[0]
                        st["seq"] += 1
                        st["dir"] = d
                        print(f"[dwbc_step] PICO flick #{st['seq']} -> {d}", flush=True)
                    else:
                        print("[dwbc_step] PICO right stick released -> stop turning", flush=True)
                    st["r_t"], st["r_hold"] = None, False
                elif now - st["r_t"] >= t_cfg["hold_s"]:
                    if not st["r_hold"]:
                        print(f"[dwbc_step] PICO right stick held -> continuous turn "
                              f"{'left' if st['r_sign'] > 0 else 'right'}", flush=True)
                    st["r_hold"] = True
                    # stick past the flick threshold = 0.3 .. 1.0 of hold_speed, the way it points now
                    mag = float(np.clip((abs(rx) - FLICK_OFF) / (1.0 - FLICK_OFF), 0.3, 1.0))
                    hold_v = mag * (1.0 if rx < 0 else -1.0) if abs(rx) > FLICK_OFF else 0.3 * st["r_sign"]
            if st["r_t"] is None and max(abs(rx), abs(ry)) < FLICK_OFF:
                st["armed_r"] = True
            out.control_data["navigate_cmd"] = [0.0, 0.0, 0.0]
        if not st["on"]:
            st["r_t"], st["r_hold"] = None, False
        out.control_data["dwbc_step"] = {"seq": st["seq"], "dir": st["dir"], "mode": st["on"], "turn_hold": hold_v}
        return out

    P._generate_unified_raw_data = gen
    print("[dwbc_step] teleop: left stick click = single-step mode, flick = one step", flush=True)
