"""Teleop-side extensions for the decoupled-WBC PICO teleop (2026-11-18). Installed by dwbc_run.py (DWBC_ABXY_RESET=1).

Slow return to the initial standing state (operator request: A+B+X+Y after the first press):
  the PICO streamer (patch_reset.py) sends control_data["reset_to_stand"] for one frame and ramps the base height
  back to standing itself. Here, in TeleopPolicy.get_action:
    * arm following is paused (A+X is ignored until the return has finished),
    * arms + waist blend from the pose published last to the initial pose (the one the teleop starts with),
      smoothstep over max(RESET_MIN_S, largest joint change / RESET_RAD_PER_S) seconds,
    * finger joints are left where they are (the grippers keep what they hold),
    * at the end the arm IK is reset to that pose, so the next A+X starts clean from it.
"""
import time

import numpy as np

from decoupled_wbc.control.policy import teleop_policy as _TP

RESET_MIN_S = 4.0          # operator 2026-11-18 19:10: hands slower and smoother
RESET_RAD_PER_S = 0.25     # average; the smoothstep peak is 1.5x (~0.37 rad/s, ~21 deg/s)
clock = time.monotonic       # tests replace it with a simulated clock

_orig_get_action = _TP.TeleopPolicy.get_action


def _get_action(self):
    rst = getattr(self, "_dwbc_reset", None)
    if rst is not None:
        self.is_active = False           # keep arm following off for the whole return (before the IK runs)
    act = _orig_get_action(self)
    if rst is not None and self.is_active:
        self.is_active = False
        print("[dwbc] A+X ignored: still returning to the standing pose", flush=True)
    now = clock()
    up = self.retargeting_ik.full_robot.get_joint_group_indices("upper_body")
    names = [self.retargeting_ik.full_robot.joint_names[i] for i in up]
    body_mask = np.array(["hand" not in n for n in names])
    if act.get("reset_to_stand") and rst is None:
        q_from = np.asarray(getattr(self, "_dwbc_last_out", act["target_upper_body_pose"]), dtype=float).copy()
        q_to = q_from.copy()
        q_to[body_mask] = np.asarray(self.retargeting_ik.full_robot.default_body_pose, dtype=float)[up][body_mask]
        dq = float(np.abs(q_to - q_from).max())
        if dq < 0.02:   # already standing (e.g. the first press): nothing to move, do not block A+X for 4 s
            print("[dwbc] already in the standing pose", flush=True)
            act["reset_to_stand"] = False
            self._dwbc_last_out = np.asarray(act["target_upper_body_pose"], dtype=float).copy()
            return act
        dur = max(RESET_MIN_S, dq / RESET_RAD_PER_S)
        rst = self._dwbc_reset = (now, dur, q_from, q_to)
        was = self.is_active
        self.is_active = False
        print(f"[dwbc] returning to the standing pose over {dur:.1f} s (largest joint change {np.degrees(dq):.0f} deg)"
              + (", arm following paused" if was else ""), flush=True)
    if rst is not None:
        t0, dur, q_from, q_to = rst
        a = min(1.0, (now - t0) / dur)
        s = a * a * (3.0 - 2.0 * a)
        out = np.asarray(act["target_upper_body_pose"], dtype=float).copy()
        out[body_mask] = (q_from + s * (q_to - q_from))[body_mask]
        act["target_upper_body_pose"] = out
        if a >= 1.0:
            self._dwbc_reset = None
            # restart the arm IK from this pose: its next output (and the next A+X) begins exactly here
            self.retargeting_ik.reset()
            # reset() puts every joint (fingers too) at the default pose: keep the fingers where they are, else the
            # grippers would open at the end of the return
            mr = self.retargeting_ik._most_recent_q
            mr[np.asarray(up)[~body_mask]] = out[~body_mask]
            print("[dwbc] back in the standing pose; A+X resumes arm following", flush=True)
    self._dwbc_last_out = np.asarray(act["target_upper_body_pose"], dtype=float).copy()
    return act


def install():
    _TP.TeleopPolicy.get_action = _get_action
