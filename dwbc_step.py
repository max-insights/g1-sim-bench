"""Single step for the decoupled-WBC leg policy (2026-11-23, operator: "a short stick push sometimes walks one step,
sometimes two - the hold time is hard to judge").

Why it happens: the leg policy switches Balance -> Walk as soon as |navigate_cmd| >= 0.05 and Walk keeps stepping in
its own learned rhythm for as long as the command stays above that (the gait clock in g1_gear_wbc_policy.py is NOT in
the observation, freq_cmd changes nothing). How many steps a stick push gives therefore depends on how long the
command stays on (stick time + chain latency) relative to that rhythm.

SingleStep closes the loop on the feet instead of on time: on a request it commands a velocity in the step direction,
watches the two feet (forward kinematics of the measured joint angles, gravity-aligned with the base orientation, so
the same code runs in sim and on the robot - the G1 has no foot contact sensors), and drops the command to zero at the
configured foot event (N-th lift-off or N-th touchdown). Then the Balance policy takes over again.

The foot detector runs all the time. For WATCH_S after the command is dropped it keeps logging lift-offs ("extra
lift": steps the policy takes on its own after the stop, e.g. squaring up a staggered stance) and then prints one
summary line per request with the lift sequence and the fore-aft foot offset at the stop and at the end.

Runs in the CONTROL loop (50 Hz), so it does not depend on the PICO -> workstation latency. Pure numpy; the caller
supplies the foot geometry (foot_pose()).
"""
import collections
import os

import numpy as np

# (vx, vy, wz) unit command per direction; turn_* (2026-11-23, operator: no turning in single-step mode) use wz rad/s
DIRS = {"fwd": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0),
        "turn_left": (0.0, 0.0, 1.0), "turn_right": (0.0, 0.0, -1.0)}
# per direction: (speed m/s, stop_on, count). MuJoCo 2026-11-23 (tests/step_sim.py): every request = 2 lifts.
# Real robot 2026-11-23 15:32 (fwd 0.20 off@touchdown#2, back 0.08 off@lift#2): the controller saw 2 lifts each time,
# but the operator saw fwd L-R-L / L-R-L-R (the policy kept stepping after the stop) and back did not move.
# -> fwd: off at the 2nd LIFT instead of the 2nd touchdown (the swing finishes on momentum, less to catch afterwards;
#    sim at 0.20: 1.5-9 cm, 0 extra lifts; at 0.15 only 0-5 cm); back: 0.15 (sim 9-16 cm; 0.08 did not move the robot).
# Override per direction in the environment: DWBC_STEP_FWD="0.15,lift,2" (speed, lift|touchdown, count), likewise
# DWBC_STEP_BACK / _LEFT / _RIGHT.
PROFILES = {"fwd": (0.20, "lift", 2), "back": (0.15, "lift", 2),
            "left": (0.20, "lift", 2), "right": (0.20, "lift", 2),
            "turn_left": (0.50, "lift", 2), "turn_right": (0.50, "lift", 2)}


# editable from the Mac app (hub dwbc_step_get / dwbc_step_set, 2026-11-23): this file overrides the defaults and the
# DWBC_STEP_* environment; the control loop re-reads it at every step request (no restart needed).
PROFILE_FILE = (os.environ.get("DWBC_STEP_PROFILE_FILE")          # tests point this elsewhere
                or os.path.join(os.path.dirname(os.path.abspath(__file__)), "step_profiles.json"))
LIMITS = {"speed": (0.05, 0.35), "turn_speed": (0.1, 1.0), "count": (1, 4), "stop_on": ("lift", "touchdown")}
DIR_CN = {"fwd": "前进", "back": "后退", "left": "左移", "right": "右移", "turn_left": "左转", "turn_right": "右转"}


def is_turn(d):
    return d.startswith("turn")


# ---- extra settings in the same file (2026-11-24, operator: single-step turning is choppy, continuous turning can't
# stop where wanted; rise speed after a squat should be adjustable from the app)
#   turn.step_deg    one right-stick flick = turn this many degrees, closed loop on the IMU yaw (flicks add up)
#   turn.hold_speed  right stick HELD in single-step mode = continuous turning up to this rate (stick-proportional);
#                    release -> the robot stops at the next lift-off once both feet have moved
#   turn.hold_s      how long the stick must be held before it counts as "held" (shorter = a flick)
#   rise.speed       base height rise rate while Y is held (upstream: 0.01 per 20 Hz frame = 0.2 m/s)
#   rise.slow_zone   the last N metres below standing height (0.74) rise at rise.slow_speed instead (0 = off)
EXTRAS = {"turn": {"step_deg": 15.0, "hold_speed": 0.5, "hold_s": 0.4},
          "rise": {"speed": 0.2, "slow_zone": 0.0, "slow_speed": 0.05}}
EXTRA_LIMITS = {"turn": {"step_deg": (5.0, 45.0), "hold_speed": (0.15, 1.0), "hold_s": (0.2, 1.0)},
                "rise": {"speed": (0.02, 0.2), "slow_zone": (0.0, 0.15), "slow_speed": (0.02, 0.2)}}
EXTRA_CN = {"turn.step_deg": "每拨一下转的角度", "turn.hold_speed": "按住时最大转速", "turn.hold_s": "按住判定时间",
            "rise.speed": "起身速度", "rise.slow_zone": "站直前减速区", "rise.slow_speed": "减速区速度"}
EXTRA_UNIT = {"turn.step_deg": "°", "turn.hold_speed": "rad/s", "turn.hold_s": "s",
              "rise.speed": "m/s", "rise.slow_zone": "m", "rise.slow_speed": "m/s"}


def validate_extras(raw):
    """{"turn": {...}, "rise": {...}} (any subset of groups / keys) -> (same shape with floats, [errors])."""
    out, errs = {}, []
    if not isinstance(raw, dict):
        return {}, ["extras 必须是对象"]
    for g, kv in raw.items():
        if g not in EXTRAS:
            errs.append(f"未知设置组 {g!r}")
            continue
        if not isinstance(kv, dict):
            errs.append(f"{g} 必须是对象")
            continue
        for k, v in kv.items():
            name = EXTRA_CN.get(f"{g}.{k}")
            if name is None:
                errs.append(f"未知设置 {g}.{k}")
                continue
            lo, hi = EXTRA_LIMITS[g][k]
            try:
                x = float(v)
            except (TypeError, ValueError):
                errs.append(f"{name}：需要数字")
                continue
            if not (lo <= x <= hi):
                errs.append(f"{name}：{x} 超出 {lo}–{hi} {EXTRA_UNIT[f'{g}.{k}']}")
                continue
            out.setdefault(g, {})[k] = round(x, 3)
    return out, errs


def merge_extras(*parts):
    out = {g: dict(v) for g, v in EXTRAS.items()}
    for p in parts:
        for g, kv in (p or {}).items():
            out.setdefault(g, {}).update(kv)
    return out


def load_extras(path=None):
    """defaults <- PROFILE_FILE["turn" / "rise"]. Returns (extras, source, errors); a bad entry ignores all extras."""
    import json
    path = path or PROFILE_FILE
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return merge_extras(), "default", []
    except (OSError, ValueError) as e:
        return merge_extras(), "default", [f"{path} 读不了：{e}"]
    got, errs = validate_extras({g: raw[g] for g in EXTRAS if g in raw})
    if errs:
        return merge_extras(), "default", errs
    return merge_extras(got), ("file" if got else "default"), []


def validate_profiles(raw):
    """{dir: {"speed", "stop_on", "count"}} (any subset of dirs) -> ({dir: (speed, stop_on, count)}, [errors])."""
    out, errs = {}, []
    if not isinstance(raw, dict):
        return {}, ["profiles 必须是对象"]
    clo, chi = LIMITS["count"]
    for d, v in raw.items():
        if d not in DIRS:
            errs.append(f"未知方向 {d!r}")
            continue
        name = DIR_CN[d]
        lo, hi = LIMITS["turn_speed" if is_turn(d) else "speed"]
        unit = "rad/s" if is_turn(d) else "m/s"
        try:
            sp, stop_on, n = float(v["speed"]), str(v["stop_on"]), int(v["count"])
        except (KeyError, TypeError, ValueError):
            errs.append(f"{name}：需要 speed（数字）、stop_on、count（整数）")
            continue
        if not (lo <= sp <= hi):
            errs.append(f"{name}：速度 {sp} 超出 {lo}–{hi} {unit}")
        elif stop_on not in LIMITS["stop_on"]:
            errs.append(f"{name}：停止条件必须是 lift（抬脚）或 touchdown（落地）")
        elif not (clo <= n <= chi):
            errs.append(f"{name}：第几次 {n} 超出 {clo}–{chi}")
        else:
            out[d] = (round(sp, 3), stop_on, n)
    return out, errs


def load_profiles(path=None, env=os.environ):
    """defaults <- DWBC_STEP_* env <- PROFILE_FILE. Returns (profiles, source, errors); a bad file is ignored as a
    whole (errors say why) so a typo can never half-apply."""
    import json
    path = path or PROFILE_FILE
    base = profiles_from_env(PROFILES, env)
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return base, "default", []
    except (OSError, ValueError) as e:
        return base, "default", [f"{path} 读不了：{e}"]
    got, errs = validate_profiles({k: v for k, v in raw.items() if k in DIRS})
    if errs:
        return base, "default", errs
    return dict(base, **got), "file", []


def save_profiles(profiles, path=None, by="", extras=None):
    """profiles {dir: (speed, stop_on, count)} (+ extras {"turn": {...}, "rise": {...}}) -> PROFILE_FILE (atomic).
    extras None keeps the extras already in the file. None / {} profiles with no extras removes the file (= defaults)."""
    import json
    import time
    path = path or PROFILE_FILE
    if extras is None and profiles:
        try:
            with open(path) as f:
                raw = json.load(f)
            extras, errs = validate_extras({g: raw[g] for g in EXTRAS if g in raw})
            if errs:
                extras = None
        except (OSError, ValueError):
            extras = None
    if not profiles and not extras:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    doc = {d: {"speed": p[0], "stop_on": p[1], "count": p[2]} for d, p in (profiles or {}).items()}
    for g, kv in (extras or {}).items():
        doc[g] = dict(kv)
    doc["updated"] = time.time()
    doc["by"] = by
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def profiles_doc(profiles):
    return {d: {"speed": p[0], "stop_on": p[1], "count": p[2]} for d, p in profiles.items()}


def profiles_from_env(base=None, env=os.environ):
    out = dict(base or PROFILES)
    for d in DIRS:
        v = env.get(f"DWBC_STEP_{d.upper()}")
        if v:
            sp, stop_on, n = (x.strip() for x in v.split(","))
            if stop_on not in ("lift", "touchdown"):
                raise ValueError(f"DWBC_STEP_{d.upper()}={v!r}: stop_on must be lift or touchdown")
            out[d] = (float(sp), stop_on, int(n))
    return out


def _rot_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat_wxyz(q):
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_pose(robot_model, q_full, base_quat_wxyz):
    """(z_left, z_right, dx) - foot heights relative to the pelvis in a gravity-aligned frame, and the fore-aft offset
    left minus right (pelvis x axis, > 0: left foot ahead). q_full: joint vector in robot_model order."""
    robot_model.cache_forward_kinematics(np.asarray(q_full, dtype=float), auto_clip=False)
    Rb = _rot_from_quat_wxyz(base_quat_wxyz)
    pl = robot_model.frame_placement(FEET[0]).translation
    pr = robot_model.frame_placement(FEET[1]).translation
    return float((Rb @ pl)[2]), float((Rb @ pr)[2]), float(pl[0] - pr[0])


def foot_heights(robot_model, q_full, base_quat_wxyz):
    zl, zr, _ = foot_pose(robot_model, q_full, base_quat_wxyz)
    return zl, zr


class SingleStep:
    """idle -> stepping (command on until the stop event or timeout) -> settle (command 0) -> idle.
    The foot detector and the after-stop watch run in every state."""
    WATCH_S = 2.0

    # turning by angle: after the command is dropped at a lift-off the swing (and the policy) keeps turning a bit.
    # COAST = that extra yaw as a fraction of the yaw of one step. MuJoCo 2026-11-24 (tests/step_sim.py turn, waist kd
    # 0): 0.5 / 0.25 stopped 4-6 deg short on 2-4 flicks, 0.0 ended within +-4 deg of 15 / 30 / 45 / 60 deg.
    COAST = 0.0
    HOLD_STALE_S = 0.5     # no hold update from the teleop for this long = released (link loss never keeps turning)
    HOLD_MIN_WZ = 0.15     # Walk needs |cmd| >= 0.05; below ~0.15 rad/s it barely turns
    # after the release: stop at the next lift-off, or this long after the release, whichever is first. Full-chain sim
    # 2026-11-24: turning in place the feet often shuffle without a 2 cm lift, and waiting for "lift #2" kept turning
    # for 3 s (+20 deg after the release)
    RELEASE_MAX_S = 0.4

    def __init__(self, speed=0.25, touchdowns=1, lift_m=0.02, land_m=0.008, timeout_s=2.0, settle_s=0.6,
                 lateral_speed=None, back_speed=None, stop_on="touchdown", profiles=None, verbose=True, turn=None):
        """stop_on "touchdown": command off at the N-th landing; "lift": off at the N-th lift-off (the swing then
        finishes on its own momentum). profiles {dir: (speed, stop_on, N)} overrides the scalar settings per dir.
        turn {"step_deg", "hold_speed"}: with a yaw in update() a turn request turns by angle (see request())."""
        self.turn = dict(EXTRAS["turn"], **(turn or {}))
        self._yaw = None           # unwrapped base yaw (rad), from update(yaw=...)
        self._target = None        # yaw target of an angle turn
        self._lift_yaw = []        # yaw at each lift-off of the current turn
        self._hold_v = 0.0
        self._hold_t = None
        self._releasing = False
        self._rel_t = None
        self._timeout = timeout_s
        self.speed = speed
        self.lateral_speed = lateral_speed if lateral_speed is not None else speed
        self.back_speed = back_speed if back_speed is not None else speed
        self.touchdowns = touchdowns
        self.stop_on = stop_on
        self.profiles = profiles
        self.lift_m, self.land_m = lift_m, land_m
        self.timeout_s, self.settle_s = timeout_s, settle_s
        self.verbose = verbose
        self.state = "idle"
        self.log = []          # (t, event, detail)
        self.summaries = []    # one dict per finished request
        self._ref = None       # foot height difference with the feet level
        self._swing = None     # "L" / "R" while a foot is in the air (detector, all states)
        self._t0 = None
        self._dir = None
        self._cmd = [0.0, 0.0, 0.0]
        self._n_td = self._n_lift = 0
        self._req = None       # summary being collected: dict(dir, t, seq, dx_stop, ...)
        self._watch_until = None
        self._hist = collections.deque(maxlen=10)   # last 0.2 s of the relative foot height (stable-landing check)

    @property
    def busy(self):
        return self.state != "idle"

    def _turn_speed(self, direction):
        if self.profiles and direction in self.profiles:
            return float(self.profiles[direction][0])
        return float(self.speed)

    def request(self, t, direction):
        if direction not in DIRS:
            raise ValueError(direction)
        if is_turn(direction) and self._yaw is not None:
            return self._request_turn(t, direction)
        if self.state != "idle":
            self._event(t, "ignored", f"{direction}: still {self.state}")
            return False
        self._finish_summary(t)              # a new request ends the previous watch window
        self.state, self._t0, self._dir = "stepping", t, direction
        self._n_td = self._n_lift = 0
        sx, sy, sw = DIRS[direction]
        if self.profiles and direction in self.profiles:
            sp, self._stop_on, self._count = self.profiles[direction]
            self._cmd = [sx * sp, sy * sp, sw * sp]
        else:
            self._stop_on, self._count = self.stop_on, self.touchdowns
            self._cmd = [sx * (self.speed if sx >= 0 else self.back_speed), sy * self.lateral_speed, sw * self.speed]
        self._req = {"dir": direction, "t": t, "seq": [], "extra": 0, "dx_stop": None, "dx_end": None,
                     "stop": None, "speed": float(max(abs(v) for v in self._cmd)), "mode": self._stop_on,
                     "yaw0": self._yaw, "target_deg": None}
        unit = "rad/s" if is_turn(direction) else "m/s"
        self._event(t, "start", f"{direction} {self._req['speed']:.2f} {unit}, off at {self._stop_on} #{self._count}")
        return True

    def _start(self, t, direction, stop_on, cmd, detail, timeout):
        self._finish_summary(t)
        self.state, self._t0, self._dir = "stepping", t, direction
        self._n_td = self._n_lift = 0
        self._stop_on, self._count = stop_on, 2
        self._cmd = list(cmd)
        self._timeout = timeout
        self._lift_yaw = []
        self._releasing = False
        self._rel_t = None
        self._req = {"dir": direction, "t": t, "seq": [], "extra": 0, "dx_stop": None, "dx_end": None,
                     "stop": None, "speed": float(max(abs(v) for v in self._cmd)), "mode": stop_on,
                     "yaw0": self._yaw, "target_deg": None}
        self._event(t, "start", f"{direction} {detail}")

    def _request_turn(self, t, direction):
        """One flick = turn step_deg, closed loop on the yaw; flicks while turning add up (the opposite way too)."""
        sign = DIRS[direction][2]
        step = np.radians(self.turn["step_deg"]) * sign
        if self.state == "stepping" and self._stop_on == "angle":
            self._target += step
            self._req["target_deg"] = float(np.degrees(self._target - self._req["yaw0"]))
            self._event(t, "add", f"{direction}: target now {self._req['target_deg']:+.0f} deg from the start")
            rem = self._target - self._yaw
            if np.sign(rem) != np.sign(self._cmd[2]) and abs(rem) > np.radians(5.0):
                # flicked back past where the robot is: turn the other way (the walk policy handles the reversal)
                self._cmd = [0.0, 0.0, -self._cmd[2]]
                self._lift_yaw = []
                self._n_lift = 0
                self._dir = "turn_left" if self._cmd[2] > 0 else "turn_right"
                self._event(t, "reverse", f"now {self._dir}, {np.degrees(abs(rem)):.0f} deg to go")
            return True
        if self.state == "stepping":
            self._event(t, "ignored", f"{direction}: still {'turning (stick held)' if self._stop_on == 'hold' else 'stepping'}")
            return False
        sp = self._turn_speed(direction)
        self._start(t, direction, "angle", [0.0, 0.0, sign * sp],
                    f"{self.turn['step_deg']:.0f} deg at {sp:.2f} rad/s (closed loop on the yaw)",
                    timeout=1.5 + 2.5 * abs(step) / sp)
        self._target = self._yaw + step
        self._req["target_deg"] = float(np.degrees(step))
        return True

    def set_hold(self, t, v):
        """Right stick held (teleop, every frame): v in [-1, 1] (> 0 = left), 0 = not held."""
        v = float(np.clip(v, -1.0, 1.0))
        self._hold_v, self._hold_t = v, t
        if v != 0.0:
            wz = np.sign(v) * max(self.HOLD_MIN_WZ, abs(v) * self.turn["hold_speed"])
            if self.state == "stepping" and self._stop_on == "hold":
                if self._releasing:
                    self._releasing = False
                    self._event(t, "hold", "stick held again")
                self._cmd = [0.0, 0.0, float(wz)]
            elif self.state == "stepping" and self._stop_on == "angle":
                self._stop_on, self._releasing = "hold", False
                self._req["mode"] = "hold"
                self._timeout = None
                self._cmd = [0.0, 0.0, float(wz)]
                self._event(t, "hold", "flick turn -> stick held: continuous")
            elif self.state == "stepping":
                return
            else:
                d = "turn_left" if wz > 0 else "turn_right"
                self._start(t, d, "hold", [0.0, 0.0, float(wz)],
                            f"held, up to {self.turn['hold_speed']:.2f} rad/s; stops at the next lift-off after release",
                            timeout=None)
        elif self.state == "stepping" and self._stop_on == "hold" and not self._releasing:
            self._release(t, "stick released")

    def _angle_done(self):
        """at a lift-off: stop now if that ends nearer the target than one more step would. Stopping still turns
        about COAST x (yaw of one step) while the swing finishes."""
        s = np.sign(self._cmd[2])
        rem = s * (self._target - self._yaw)
        if len(self._lift_yaw) >= 2:
            per = abs(self._lift_yaw[-1] - self._lift_yaw[-2])
        else:
            per = abs(self._cmd[2]) * 0.35
        per = max(per, np.radians(2.0))
        coast = self.COAST * per
        return abs(rem - coast) <= abs(rem - coast - per)

    def _release(self, t, why):
        self._rel_t = t
        self._releasing = True
        self._event(t, "release", f"{why}: stop at the next lift-off (at most {self.RELEASE_MAX_S:.1f} s)")

    def cancel(self, t, why="cancel"):
        if self.state == "stepping":
            self._stop(t, f"cancel ({why})", None)

    def update(self, t, z_left, z_right, dx=None, yaw=None):
        """Returns the navigate command override [vx, vy, wz] while stepping / settling, else None.
        yaw: base yaw (rad) - enables turning by angle."""
        if yaw is not None:
            if self._yaw is None:
                self._yaw = float(yaw)
            else:
                self._yaw += (float(yaw) - self._yaw + np.pi) % (2 * np.pi) - np.pi
        dz = z_left - z_right           # > 0: left foot higher
        if self._ref is None:
            self._ref = dz
        if self.state == "idle" and self._swing is None and abs(dz - self._ref) < 0.01:
            # track the standing offset slowly, only with the feet about level
            self._ref = 0.98 * self._ref + 0.02 * dz
        rel = dz - self._ref
        self._hist.append(rel)
        lift = land = None
        if self._swing is None:
            if rel > self.lift_m:
                lift = self._swing = "L"
            elif rel < -self.lift_m:
                lift = self._swing = "R"
        elif abs(rel) < self.land_m:
            land, self._swing = self._swing, None
        elif (len(self._hist) == self._hist.maxlen and max(self._hist) - min(self._hist) < 0.003
              and abs(rel) < self.lift_m):
            # the foot has stopped moving below the lift threshold but not back to the old level (the stance changed,
            # e.g. a staggered stance after a step): count it as landed and take the new level as the reference
            land, self._swing = self._swing, None
            self._ref = dz

        out = None
        if self.state == "stepping":
            if self._stop_on == "hold" and not self._releasing and (self._hold_t is None
                                                                     or t - self._hold_t > self.HOLD_STALE_S):
                self._release(t, "no stick update")
            if lift:
                self._n_lift += 1
                self._req["seq"].append(lift)
                yaw_s = "" if self._yaw is None else f" yaw {np.degrees(self._yaw - (self._req['yaw0'] or 0)):+.1f} deg"
                self._event(t, "lift", f"{lift} #{self._n_lift}{yaw_s if is_turn(self._dir) else ''}")
                if self._stop_on == "lift" and self._n_lift >= self._count:
                    self._stop(t, f"lift #{self._n_lift}", dx)
                elif self._stop_on == "angle":
                    self._lift_yaw.append(self._yaw)
                    if self._n_lift >= 2 and self._angle_done():
                        self._stop(t, f"lift #{self._n_lift}, angle reached", dx)
                elif self._stop_on == "hold" and self._releasing:
                    self._stop(t, f"lift #{self._n_lift} after release", dx)
            elif land:
                self._n_td += 1
                self._event(t, "touchdown", f"{land} #{self._n_td}")
                if self._stop_on == "touchdown" and self._n_td >= self._count:
                    self._stop(t, f"touchdown #{self._n_td}", dx)
            if self.state == "stepping" and self._stop_on == "angle":
                s = np.sign(self._cmd[2])
                if s * (self._target - self._yaw) <= 0:          # already there / overshot (or flicked back)
                    self._stop(t, "angle passed", dx)
            limit = self.timeout_s if self._stop_on in ("lift", "touchdown") else self._timeout
            if self._stop_on == "hold" and self._releasing:
                limit = None
                if self.state == "stepping" and t - self._rel_t >= self.RELEASE_MAX_S:
                    self._stop(t, f"{self.RELEASE_MAX_S:.1f} s after release", dx)
            if self.state == "stepping" and limit is not None and t - self._t0 > limit:
                self._event(t, "timeout", f"{self._n_lift} lifts, {self._n_td} touchdowns")
                self._stop(t, "timeout", dx)
            out = list(self._cmd) if self.state == "stepping" else [0.0, 0.0, 0.0]
        else:
            if self._req is not None and self._watch_until is not None and t <= self._watch_until and lift:
                self._req["extra"] += 1
                self._req["seq"].append(lift.lower())
                self._event(t, "extra lift", f"{lift} ({t - self._req['stop']:.2f} s after the stop)")
            if self.state == "settle":
                if t - self._t0 >= self.settle_s:
                    self.state = "idle"
                    self._event(t, "idle", "")
                else:
                    out = [0.0, 0.0, 0.0]
        if self._req is not None and self._watch_until is not None:
            if dx is not None:
                self._req["dx_end"] = dx
            if t > self._watch_until and self._swing is None:
                self._finish_summary(t)
        return out

    def _stop(self, t, why, dx):
        self.state, self._t0 = "settle", t
        self._req["stop"] = t
        self._req["why"] = why
        self._req["dx_stop"] = dx
        self._req["yaw_stop"] = self._yaw
        self._watch_until = t + self.WATCH_S

    def _finish_summary(self, t):
        r = self._req
        if r is None or r.get("stop") is None:
            return
        self._req, self._watch_until = None, None
        turned = ""
        if r.get("yaw0") is not None and self._yaw is not None:
            r["dyaw_stop_deg"] = float(np.degrees(r["yaw_stop"] - r["yaw0"])) if r.get("yaw_stop") is not None else None
            r["dyaw_deg"] = float(np.degrees(self._yaw - r["yaw0"]))
            if is_turn(r["dir"]):
                tgt = "" if r.get("target_deg") is None else f" (target {r['target_deg']:+.0f})"
                turned = f"; turned {r['dyaw_deg']:+.1f} deg{tgt}, {r['dyaw_stop_deg']:+.1f} at the stop"
        self.summaries.append(r)
        f = lambda v: "-" if v is None else f"{100 * v:+.1f} cm"   # noqa: E731
        seq = "-".join(r["seq"]) or "none"
        self._event(t, "summary", f"{r['dir']}: feet {seq} (lower case = after the stop, {r['extra']} extra); "
                                  f"stopped at {r['why']}; left-right fore-aft offset at stop {f(r['dx_stop'])}, "
                                  f"{self.WATCH_S:.0f} s later {f(r['dx_end'])}{turned}")

    def _event(self, t, ev, detail):
        self.log.append((t, ev, detail))
        if self.verbose:
            print(f"[dwbc_step] {t:8.2f} {ev} {detail}", flush=True)
