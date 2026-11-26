"""Run / episode scorecard for the DWBC teleop chain (2026-11-26).

Six dimensions, each a 0-100 score (higher = better) plus the measured value behind it, so two runs (or two
recordings) can be compared at a glance: where it got better, where it got worse.

  latency  时延        end-to-end PICO motion -> robot joint at the target, p50 ms (dwbc_latency blocks)
  stall    卡顿        share of frames > 100 ms above the usual (dwbc_latency stall) ; teleop period p99
  jitter   抖动        measured arm joints while the operator's hands hold still (0.5 s windows, both wrist targets
                       move < 1.5 cm): joint std, median over windows, deg
  jumps    突变        per arm-follow minute: IK-output jerk (joint speed changes > 2 rad/s between frames), measured
                       arm swings (> 25 deg in 0.5 s while the hands move < 5 cm), outward swings (shoulder > 15 deg)
  walking  走路流畅度   per step (single-step events in control.log, else 1.5 s windows of continuous walking): pelvis
                       roll peak-to-peak, roll rate max, pitch peak-to-peak, extra steps; torso IMU when recorded
  ik       逆解        arm IK time per frame p50 / p99 (lat_teleop.csv), frames over 50 ms

Inputs (all in the run dir written by dwbc_latency.py): lat_control.csv, lat_track.csv, lat_teleop.csv, teleop_pose.csv,
control.log, lat_body.csv (runs from 2026-11-26 on). Walking for older runs comes from the g1_hub episodes recorded
during the run (lowstate IMU at 1 kHz). An episode is scored on its own time window of the run.

The anchors (value -> score 100 / value -> score 0) are set from the real runs of 11-20 .. 11-26 (see ANCHORS); a
score is linear in between and clamped. A dimension without data (no walking, no arm-follow) is null, never 0.

    python dwbc_score.py run   [run dir]           # one run (default real_latest)
    python dwbc_score.py ep    <episode dir>       # one g1_hub episode
    python dwbc_score.py table [N]                 # the last N runs, one line each
Needs numpy (g1_hub's conda env has it).
"""
import json
import math
import os
import re
import sys

import numpy as np

VERSION = 1
LOGS = os.path.expanduser(os.environ.get("DWBC_LOGS", "~/piper_ws/dwbc_logs"))
DATA = os.environ.get("G1_DATA", "/data/robot_data/g1/g1_data")

# value at score 100, value at score 0 (lower is better for all of them)
ANCHORS = {
    "e2e_ms": (160.0, 320.0),
    "stall_pct": (0.0, 20.0),
    "hold_std_deg": (0.2, 1.2),
    "jerk_min": (0.0, 20.0),
    "swing_min": (0.0, 6.0),
    "outward_min": (0.0, 8.0),
    "roll_pp_deg": (5.0, 16.0),
    "roll_rate_dps": (25.0, 80.0),
    "pitch_pp_deg": (3.0, 12.0),
    "extra_pct": (0.0, 30.0),
    "ik_p50_ms": (5.0, 60.0),
    "ik_p99_ms": (20.0, 120.0),
}
DIMS = [  # key, label, what the headline value is
    ("latency", "时延", "端到端延时 p50"),
    ("stall", "卡顿", "卡顿帧占比"),
    ("jitter", "抖动", "手停住时关节抖动"),
    ("jumps", "突变", "关节突变"),
    ("walking", "走路流畅度", "每步骨盆侧晃"),
    ("ik", "逆解", "逆解耗时 p50"),
]
# upper-body order of teleop_pose.csv q0..q30 (DWBC robot model): waist 3, left arm 7, left hand 7, right arm 7, ...
ARM_Q = list(range(3, 10)) + list(range(17, 24))
# lat_track.csv meas0..13 = robot_model "arms": left 7 then right 7 (shoulder pitch, roll, yaw, elbow, wrist r/p/y)
OUTWARD = [(1, 1), (2, 1), (8, -1), (9, -1)]       # (index, sign): shoulder roll / yaw moving away from the body
MIN_ARM_MIN = 0.3                                  # less arm-follow than this (minutes): arm dimensions are null


# ------------------------------------------------------------------------------------------------ io
def read_csv(path, cols=None):
    """{column: float array} (blank -> nan). cols limits what is parsed."""
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as f:
        head = f.readline().strip().split(",")
        idx = [i for i, c in enumerate(head) if cols is None or c in cols]
        rows = []
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) != len(head):
                continue                            # a row being written
            rows.append([float(p[i]) if p[i] not in ("", "None", "True", "False") else
                         (1.0 if p[i] == "True" else 0.0 if p[i] == "False" else math.nan) for i in idx])
    a = np.array(rows, dtype=float).reshape(-1, len(idx))
    return {head[i]: a[:, k] for k, i in enumerate(idx)}


def load_stream(ep, name):
    p = os.path.join(ep, "streams", name)
    if not os.path.isfile(p + ".bin"):
        return None
    d = json.load(open(p + ".dtype.json"))["dtype"]
    dt = np.dtype({"names": d["names"], "formats": d["formats"]})
    if os.path.getsize(p + ".bin") < dt.itemsize:
        return None
    return np.memmap(p + ".bin", dtype=dt, mode="r")


def step_events(run):
    """[(t_start, t_idle, direction, extra steps or None)] from control.log (dwbc_step, CLOCK_MONOTONIC s)."""
    ev, cur, extra = [], None, {}
    try:
        f = open(os.path.join(run, "control.log"), errors="replace")
    except OSError:
        return ev
    for line in f:
        m = re.search(r"\[dwbc_step\] ([0-9.]+) (start|idle)\s*(\S*)", line)
        if m:
            t, w, a = float(m.group(1)), m.group(2), m.group(3)
            if w == "start":
                cur = [t, a]
            elif cur:
                ev.append([cur[0], t, cur[1], None])
                cur = None
            continue
        m = re.search(r"\[dwbc_step\] ([0-9.]+) summary (\S+):.*\(lower case = after the stop, ([0-9]+) extra\)", line)
        if m:
            extra[float(m.group(1))] = int(m.group(3))
    for e in ev:
        near = [v for k, v in extra.items() if e[0] <= k <= e[1] + 3]
        e[3] = near[0] if near else None
    return [tuple(e) for e in ev]


# ------------------------------------------------------------------------------------------------ helpers
def lin(v, key):
    if v is None or not np.isfinite(v):
        return None
    good, bad = ANCHORS[key]
    return float(np.clip(100.0 * (bad - v) / (bad - good), 0.0, 100.0))


def comb(*scores):
    """A dimension's score: the worst of its parts (one bad part should show)."""
    s = [x for x in scores if x is not None]
    return round(min(s)) if s else None


def r(v, nd=1):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def mask_window(t, win):
    return np.ones(len(t), bool) if win is None else (t >= win[0]) & (t <= win[1])


def swings(t, q, tg, deg, cols):
    c, k, n = 0, 0, len(t)
    while k < n:
        j = k
        while j + 1 < n and t[j + 1] - t[k] <= 0.5 and t[j + 1] - t[j] < 0.15:
            j += 1
        if j > k and np.abs(q[j, cols] - q[k, cols]).max() > math.radians(deg) and \
                max(np.linalg.norm(tg[j, :3] - tg[k, :3]), np.linalg.norm(tg[j, 7:10] - tg[k, 7:10])) < 0.05:
            c, k = c + 1, j
            continue
        k += 1
    return c


def outward(t, q, tg):
    c, k, n = 0, 0, len(t)
    while k < n:
        j = k
        while j + 1 < n and t[j + 1] - t[k] <= 0.5 and t[j + 1] - t[j] < 0.15:
            j += 1
        if j > k and max(np.linalg.norm(tg[j, :3] - tg[k, :3]), np.linalg.norm(tg[j, 7:10] - tg[k, 7:10])) < 0.06 \
                and max(s * (q[j, i] - q[k, i]) for i, s in OUTWARD) > math.radians(15):
            c, k = c + 1, j
            continue
        k += 1
    return c


def follow_lags(tt, cmd, meas, win_s=5.0, max_lag=40):
    """Robot follow lag (ms) per 5 s window, same method as dwbc_latency.follow_lag (numpy)."""
    out = []
    if len(tt) < 60:
        return out
    dt = (tt[-1] - tt[0]) / (len(tt) - 1)
    n = int(round(win_s / dt))
    for a in range(0, len(tt) - 60, n):
        c, m = cmd[a:a + n], meas[a:a + n]
        lags = []
        for j in range(c.shape[1]):
            cj, mj = c[:, j], m[:, j]
            if not (np.isfinite(cj).all() and np.isfinite(mj).all()) or np.ptp(cj) < 0.05:
                continue
            L = min(max_lag, len(cj) // 3)
            e = [np.mean(((cj[:len(cj) - k] - cj[:len(cj) - k].mean()) - (mj[k:] - mj[k:].mean())) ** 2)
                 for k in range(L + 1)]
            lags.append(int(np.argmin(e)) * dt * 1e3)
        if lags:
            out.append(float(np.median(lags)))
    return out


def quat_rp(q):
    """wxyz -> roll, pitch (rad)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


# ------------------------------------------------------------------------------------------------ dimensions
def dim_latency(lat_mod, run, win, lat_summary):
    """e2e p50 + stall share; whole run from the hub's cached summary, a window computed here."""
    if lat_summary is None and lat_mod is not None:
        ctrl, track, pico_min = lat_mod.read_run(run)
        f = lat_mod._f
        if win is not None:
            ctrl = [x for x in ctrl if win[0] <= (f(x.get("t_read")) or f(x.get("t_pick")) or 0) <= win[1]]
            track = [x for x in track if win[0] <= (f(x.get("t")) or 0) <= win[1]]
        if track:
            tt = np.array([f(x["t"]) for x in track])
            cm = np.array([[f(x[f"cmd{i}"]) for i in range(14)] for x in track], float)
            ms = np.array([[f(x[f"meas{i}"]) for i in range(14)] for x in track], float)
            lv = follow_lags(tt, cm, ms)
        else:
            lv = []
        lat_summary = lat_mod.summary(ctrl, track, pico_min, None, lag_values=lv or None)
    if not lat_summary or not lat_summary.get("frames"):
        return None, None
    e2e = lat_summary.get("e2e_ms")
    st = lat_summary.get("stall") or {}
    lat = {"score": comb(lin(e2e, "e2e_ms")), "value": e2e, "unit": "ms",
           "detail": [{"label": "最大的一段", "value": f"{lat_summary.get('worst_label') or '—'} "
                                                    f"{lat_summary.get('worst_ms') or '—'} ms"},
                      {"label": "机器人跟随", "value": f"{lat_summary.get('follow_lag_ms') or '—'} ms"}]}
    stall = {"score": comb(lin(st.get("pct"), "stall_pct")), "value": st.get("pct"), "unit": "%",
             "detail": [{"label": "最长一次多出", "value": f"{st.get('worst_extra_ms') or '—'} ms"},
                        {"label": "主要来源", "value": st.get("cause_label") or "—"}]}
    return lat, stall


def dim_arm(run, win, pose, track):
    """jitter / jumps from teleop_pose.csv (IK output, wrist targets) and lat_track.csv (measured arm joints)."""
    if pose is None or "t_mono_ns" not in pose:
        return None, None, 0.0
    t = pose["t_mono_ns"] / 1e9
    act = (pose["active"] == 1) & mask_window(t, win)
    tg = np.stack([pose[f"wrist_{s}{i}"] for s in "lr" for i in range(7)], 1)
    q = np.stack([pose[f"q{i}"] for i in ARM_Q], 1)
    ok = act & np.isfinite(tg).all(1) & np.isfinite(q).all(1)
    ta, tga, qa = t[ok], tg[ok], q[ok]
    mins = len(ta) / 1200.0
    if mins < MIN_ARM_MIN:
        return None, None, mins
    dt = np.diff(ta)
    good = (dt > 0.02) & (dt < 0.15)
    vs = np.diff(qa, axis=0) / np.maximum(dt, 1e-3)[:, None]
    jerk = int((good[1:] & good[:-1] & (np.abs(np.diff(vs, axis=0)).max(1) > 2.0)).sum()) / mins
    cap = int((good & (np.abs(vs).max(1) > 3.0)).sum()) / mins
    sw = ow = hold = hold90 = None
    if track is not None and len(track["t"]) > 100:
        lt = track["t"]
        me = np.stack([track[f"meas{i}"] for i in range(14)], 1)
        qm = np.stack([np.interp(ta, lt, me[:, i]) for i in range(14)], 1)
        sw = swings(ta, qm, tga, 25, list(range(14))) / mins
        ow = outward(ta, qm, tga) / mins
        # jitter: 0.5 s windows of the 50 Hz measurement where both wrist targets stay within 1.5 cm
        stds = []
        k = 0
        while k < len(ta):
            j = np.searchsorted(ta, ta[k] + 0.5)
            if j - k >= 6 and ta[j - 1] - ta[k] > 0.4 and np.all(np.diff(ta[k:j]) < 0.15):
                d = max(np.ptp(tga[k:j, 0:3], axis=0).max(), np.ptp(tga[k:j, 7:10], axis=0).max())
                if d < 0.015:
                    w = (lt >= ta[k]) & (lt <= ta[j - 1])
                    if w.sum() >= 15:
                        stds.append(np.degrees(np.std(me[w], axis=0).max()))
            k = j
        if len(stds) >= 5:
            hold, hold90 = float(np.median(stds)), float(np.percentile(stds, 90))
    jitter = None
    if hold is not None:
        jitter = {"score": comb(lin(hold, "hold_std_deg")), "value": r(hold, 2), "unit": "°",
                  "detail": [{"label": "90 分位", "value": f"{r(hold90, 2)}°"},
                             {"label": "手停住的时段", "value": f"{len(stds)} 段 × 0.5 s"}]}
    jumps = {"score": comb(lin(jerk, "jerk_min"), lin(sw, "swing_min"), lin(ow, "outward_min")),
             "value": r(jerk), "unit": "次/分钟",
             "detail": [{"label": "实测大摆动 >25°", "value": f"{r(sw)} 次/分钟" if sw is not None else "—"},
                        {"label": "实测往外甩 >15°", "value": f"{r(ow)} 次/分钟" if ow is not None else "—"},
                        {"label": "速度顶上限的帧", "value": f"{r(cap)} 次/分钟"}]}
    return jitter, jumps, mins


def dim_ik(win, teleop):
    if teleop is None or "ik_ms" not in teleop:
        return None
    t = teleop["t_read"]
    m = (teleop["active"] == 1) & mask_window(t, win) & np.isfinite(teleop["ik_ms"])
    x = teleop["ik_ms"][m]
    if len(x) < 100:
        return None
    p50, p99 = float(np.percentile(x, 50)), float(np.percentile(x, 99))
    over = 100.0 * float((x > 50).mean())
    per = teleop["period_ms"][m]
    per = per[np.isfinite(per)]
    return {"score": comb(lin(p50, "ik_p50_ms"), lin(p99, "ik_p99_ms")), "value": r(p50), "unit": "ms",
            "detail": [{"label": "99 分位", "value": f"{r(p99)} ms"},
                       {"label": "超过 50 ms 的帧", "value": f"{r(over, 2)}%"},
                       {"label": "teleop 周期 99 分位", "value": f"{r(np.percentile(per, 99)) if len(per) else '—'} ms"}]}


def walk_windows(run, win, pose):
    """(a, b, label, extra) windows to score: single steps from control.log, else 1.5 s chunks of continuous walking
    (navigate command |v| > 0.05 in teleop_pose.csv)."""
    ev = [e for e in step_events(run) if win is None or (win[0] <= e[0] and e[1] <= win[1])]
    out = [(a, b + 0.8, d, x) for a, b, d, x in ev]
    if pose is not None and "nav0" in pose:
        t = pose["t_mono_ns"] / 1e9
        nav = np.stack([pose["nav0"], pose["nav1"], pose["nav2"]], 1)
        mv = (np.nan_to_num(np.abs(nav)).max(1) > 0.05) & mask_window(t, win)
        busy = [(a, b) for a, b, _, _ in out]
        k = 0
        while k < len(t):
            if not mv[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(t) and mv[j + 1] and t[j + 1] - t[j] < 0.3:
                j += 1
            a = t[k]
            while a + 1.5 <= t[j]:
                if not any(x <= a + 1.5 and a <= y for x, y in busy):
                    out.append((a, a + 1.5, "walk", None))
                a += 1.5
            k = j + 1
    return out


def body_source(run, win, episodes):
    """(t, roll, pitch, roll_rate, pitch_rate, torso_roll or None) from lat_body.csv, else from the episodes'
    lowstate IMU. None when neither covers the window."""
    b = read_csv(os.path.join(run, "lat_body.csv"))
    if b is not None and len(b["t"]) > 50:
        roll, pitch = quat_rp(np.stack([b[f"pq{i}"] for i in range(4)], 1))
        tr = None
        if np.isfinite(b["tq0"]).all():
            tr = quat_rp(np.stack([b[f"tq{i}"] for i in range(4)], 1))[0]
        return b["t"], roll, pitch, b["pg0"], b["pg1"], tr
    parts = []
    for ep in episodes:
        ls = load_stream(ep, "lowstate")
        if ls is None:
            continue
        t = ls["t_mono_ns"].astype(np.float64) / 1e9
        if win is not None and (t[-1] < win[0] or t[0] > win[1]):
            continue
        rpy, g = np.asarray(ls["imu_rpy"], float), np.asarray(ls["imu_gyro"], float)
        s = slice(None, None, 5)                                   # 1 kHz -> 200 Hz is plenty
        parts.append((t[s], rpy[s, 0], rpy[s, 1], g[s, 0], g[s, 1]))
    if not parts:
        return None
    cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    o = np.argsort(cat[0])
    return tuple(c[o] for c in cat) + (None,)


def dim_walking(run, win, pose, episodes):
    wins = walk_windows(run, win, pose)
    if not wins:
        return None
    src = body_source(run, win, episodes)
    if src is None:
        return {"score": None, "value": None, "unit": "°", "steps": len(wins),
                "detail": [{"label": "说明", "value": f"走了 {len(wins)} 段，但这次运行没有姿态数据（没录制 episode）"}]}
    t, roll, pitch, rr, pr, troll = src
    R = {"roll": [], "pitch": [], "rate": [], "prate": [], "torso": [], "extra": []}
    for a, b, d, x in wins:
        w = (t >= a) & (t <= b)
        if w.sum() < 20:
            continue
        R["roll"].append(np.degrees(np.ptp(roll[w])))
        R["pitch"].append(np.degrees(np.ptp(pitch[w])))
        R["rate"].append(np.degrees(np.abs(rr[w]).max()))
        R["prate"].append(np.degrees(np.abs(pr[w]).max()))
        if troll is not None:
            R["torso"].append(np.degrees(np.ptp(troll[w])))
        if x is not None:
            R["extra"].append(x > 0)
    n = len(R["roll"])
    if n == 0:
        return {"score": None, "value": None, "unit": "°", "steps": len(wins),
                "detail": [{"label": "说明", "value": "走路的时段没有姿态数据覆盖"}]}
    med = {k: float(np.median(v)) for k, v in R.items() if v and k != "extra"}
    extra = 100.0 * float(np.mean(R["extra"])) if R["extra"] else None
    nstep = sum(1 for w in wins if w[2] != "walk")
    det = [{"label": "骨盆侧晃角速度", "value": f"{r(med['rate'])}°/s"},
           {"label": "骨盆前后晃", "value": f"{r(med['pitch'])}°"},
           {"label": "多迈一步", "value": f"{sum(R['extra'])} / {len(R['extra'])}" if R["extra"] else "—"},
           {"label": "样本", "value": f"单步 {nstep} 次，连续走 {n - nstep if n >= nstep else 0} 段 × 1.5 s"}]
    # (torso IMU is recorded in lat_body.csv but not shown yet: its roll mixes with waist yaw while the torso leans)
    return {"score": comb(lin(med["roll"], "roll_pp_deg"), lin(med["rate"], "roll_rate_dps"),
                          lin(med["pitch"], "pitch_pp_deg"), lin(extra, "extra_pct")),
            "value": r(med["roll"]), "unit": "°", "steps": n, "detail": det}


# ------------------------------------------------------------------------------------------------ entry points
def run_episodes(run):
    """g1_hub episodes recorded during this run (meta.dwbc_run_dirs)."""
    out = []
    rr = os.path.realpath(run)
    try:
        sets = [os.path.join(DATA, d) for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))]
    except OSError:
        return out
    for s in sets:
        for e in sorted(os.listdir(s)):
            m = os.path.join(s, e, "meta.json")
            if not e.startswith("ep_") or not os.path.isfile(m):
                continue
            try:
                meta = json.load(open(m))
            except (OSError, ValueError):
                continue
            dirs = meta.get("dwbc_run_dirs") or [((meta.get("dwbc") or {}).get("run_dir"))]
            if any(d and os.path.realpath(d) == rr for d in dirs):
                out.append(os.path.join(s, e))
    return out


def wall_to_mono(pose):
    """offset (s) to add to a wall-clock time to get CLOCK_MONOTONIC, from teleop_pose.csv."""
    if pose is None or "t_wall_ns" not in pose or not len(pose["t_wall_ns"]):
        return None
    return float(np.median(pose["t_mono_ns"] - pose["t_wall_ns"])) / 1e9


def score(run, win=None, lat_mod=None, lat_summary=None, episodes=None):
    """The scorecard of a run (win None) or of a window [mono a, mono b] of it."""
    pose = read_csv(os.path.join(run, "teleop_pose.csv"))
    track = read_csv(os.path.join(run, "lat_track.csv"))
    teleop = read_csv(os.path.join(run, "lat_teleop.csv"), {"t_read", "ik_ms", "period_ms", "active"})
    if episodes is None:
        episodes = run_episodes(run)
    lat, stall = dim_latency(lat_mod, run, win, lat_summary)
    jitter, jumps, mins = dim_arm(run, win, pose, track)
    walking = dim_walking(run, win, pose, episodes)
    ik = dim_ik(win, teleop)
    got = {"latency": lat, "stall": stall, "jitter": jitter, "jumps": jumps, "walking": walking, "ik": ik}
    dims = []
    for k, label, what in DIMS:
        d = got.get(k) or {"score": None, "value": None, "unit": None, "detail": []}
        dims.append(dict(d, key=k, label=label, what=what, grade=grade(d.get("score"))))
    sc = [d["score"] for d in dims if d["score"] is not None]
    return {"version": VERSION, "run": os.path.basename(run), "window": list(win) if win else None,
            "arm_follow_min": r(mins), "overall": round(sum(sc) / len(sc)) if sc else None, "dims": dims}


def grade(s):
    if s is None:
        return None
    return "优" if s >= 85 else "良" if s >= 70 else "中" if s >= 50 else "差"


def score_episode(ep, lat_mod=None):
    meta = json.load(open(os.path.join(ep, "meta.json")))
    dirs = meta.get("dwbc_run_dirs") or [((meta.get("dwbc") or {}).get("run_dir"))]
    run = next((d for d in dirs if d and os.path.isdir(d)), None)
    if run is None or not meta.get("t_start_ns") or not meta.get("t_stop_ns"):
        return None
    pose = read_csv(os.path.join(run, "teleop_pose.csv"), {"t_wall_ns", "t_mono_ns"})
    off = wall_to_mono(pose)
    if off is None:
        return None
    win = (meta["t_start_ns"] / 1e9 + off, meta["t_stop_ns"] / 1e9 + off)
    out = score(run, win, lat_mod=lat_mod, episodes=[ep])
    out.update(episode=os.path.basename(ep), dataset=os.path.basename(os.path.dirname(ep)), task=meta.get("task"),
               start=meta["t_start_ns"] / 1e9, duration_s=r(meta.get("duration_s")))
    return out


def _lat_mod():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dwbc_latency.py")
    spec = importlib.util.spec_from_file_location("dwbc_latency", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def render(s):
    head = s.get("task") or s["run"]
    print(f"== {head}  arm-follow {s['arm_follow_min']} min  overall {s['overall']}")
    for d in s["dims"]:
        v = "—" if d["value"] is None else f"{d['value']} {d['unit']}"
        sc = "—" if d["score"] is None else f"{d['score']:3d} {d['grade']}"
        print(f"   {d['label']:<6s} {sc:>7s}   {d['what']} {v}   " +
              "  ".join(f"{x['label']} {x['value']}" for x in d["detail"]))


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "run"
    mod = _lat_mod()
    if cmd == "run":
        d = os.path.realpath(a[1] if len(a) > 1 else os.path.join(LOGS, "real_latest"))
        render(score(d, lat_mod=mod, lat_summary=mod.run_summary(d, series=False)))
    elif cmd == "ep":
        s = score_episode(a[1], mod)
        render(s) if s else print("no DWBC run / window for this episode")
    elif cmd == "table":
        n = int(a[1]) if len(a) > 1 else 10
        for x in mod.list_runs(LOGS, n):
            d = os.path.join(LOGS, x["run"])
            s = score(d, lat_mod=mod, lat_summary=mod.run_summary(d, series=False))
            print(f"{x['run']}  " + "  ".join(f"{d['label']}={d['value']}/{d['score']}" for d in s["dims"]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
