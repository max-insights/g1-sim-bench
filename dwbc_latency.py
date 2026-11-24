"""Per-hop latency monitor for the decoupled-WBC PICO teleop chain (2026-11-20).

Control chain (all stamps on the workstation's CLOCK_MONOTONIC, shared by both processes, except the PICO's own clock):

  PICO headset --Wi-Fi--> xrobotoolkit-pc-service (:63901) --SDK--> teleop process (dwbc_run.py teleop, 20 Hz)
      [1] PICO sample -> read by teleop    pico_streamer reads the latest pose; age = wall clock - PICO timestamp,
                                           shown above its minimum (the two clocks are not synchronised)
      [2] teleop period                    time between two reads (the loop is 20 Hz = 50 ms when on time)
      [3] streamer                         TeleopStreamer.get_streamer_data (SDK read + pose processing)
      [4] arm IK                           TeleopRetargetingIK.get_action (body IK, multi-start when DWBC_IK_GLOBAL=1)
      [5] teleop other                     rest of TeleopPolicy.get_action (activation, A+B+X+Y return, ...)
      [6] publish                          msgpack + ROS 2 publish of the goal (CONTROL_GOAL_TOPIC)
  --ROS 2 (domain 77, loopback)--> control process (dwbc_run.py control, 50 Hz)
      [7] ROS transport                    publish -> subscriber callback
      [8] wait for control loop            callback -> the 50 Hz loop picks the message up
      [9] control -> lowcmd                pick-up -> G1Env.queue_action wrote rt/lowcmd (policy + safety)
      [10] interpolation                   pick-up -> time the interpolated arm trajectory reaches this target
                                           (target_time = publish + 50 ms, longer when the joint speed cap binds)
  --DDS rt/lowcmd over enp13s0--> G1 motors
      [11] robot follow lag                measured arm joints vs commanded arm joints (cross-correlation)

Recording (inside the two processes, installed by dwbc_run.py when DWBC_LATENCY != "0"):
  <run dir>/lat_teleop.csv    one row per teleop frame
  <run dir>/lat_control.csv   one row per goal the control loop picked up (the whole chain [1]-[10] in one row)
  <run dir>/lat_track.csv     one row per control cycle: commanded and measured arm joints (for [11])
  The run dir is $DWBC_LAT_DIR, else the directory of the process's stdout log (dwbc_real.sh: real_<stamp>/).
  The instrumentation never changes what the loops compute and never raises into them: on any error it prints
  once and switches itself off.

Live view / summary (stdlib only, any python3):
  python3 ~/piper_ws/dwbc/dwbc_latency.py watch [run dir]      # refreshes every second, last 5 s
  python3 ~/piper_ws/dwbc/dwbc_latency.py report [run dir]     # whole run, also writes lat_summary.txt
  default run dir: ~/piper_ws/dwbc_logs/real_latest
"""
import collections
import os
import sys
import time

# ------------------------------------------------------------------------------------------------ instrumentation
TELEOP_COLS = ["seq", "t_read", "pico_ts_ns", "wall_ns", "pico_new", "period_ms", "streamer_ms", "ik_ms",
               "action_ms", "pub_ms", "active"]
CONTROL_COLS = ["seq", "t_read", "pico_ts_ns", "wall_ns", "pico_new", "period_ms", "streamer_ms", "ik_ms",
                "action_ms", "t_pub", "t_arrive", "t_pick", "t_target", "t_waypoint", "t_sent", "active"]
N_ARM = 14

_on = True
_files = {}
_last_flush = 0.0


def _off(where, e):
    global _on
    if _on:
        _on = False
        print(f"[dwbc_latency] disabled after an error in {where}: {e!r} (control is not affected)", flush=True)


def _safe(where, fn, *a):
    if not _on:
        return None
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 - instrumentation must never break the loops
        _off(where, e)
        return None


def _run_dir():
    d = os.environ.get("DWBC_LAT_DIR")
    if not d:
        try:
            p = os.readlink("/proc/self/fd/1")
            if os.path.isfile(p):
                d = os.path.dirname(p)
        except OSError:
            pass
    if not d or not os.access(d, os.W_OK):
        d = os.path.join(os.environ.get("DWBC_LOGS", os.path.expanduser("~/piper_ws/dwbc_logs")),
                         "lat_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(d, exist_ok=True)
    return d


def _open(name, cols):
    path = os.path.join(_run_dir(), name)
    f = open(path, "a", buffering=1 << 16)
    if f.tell() == 0:
        f.write(",".join(cols) + "\n")
    _files[name] = f
    print(f"[dwbc_latency] recording {path}", flush=True)
    return f


def _write(name, line):
    global _last_flush
    _files[name].write(line + "\n")
    now = time.monotonic()
    if now - _last_flush > 0.5:
        _last_flush = now
        for f in _files.values():
            f.flush()


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return f"{v:.6f}"


def _wrap(cls, name, pre, post):
    """cls.name runs exactly as before; pre(self, *a) / post(self, ctx, result) are guarded."""
    orig = getattr(cls, name)

    def wrapped(self, *a, **k):
        ctx = _safe(f"{cls.__name__}.{name}", pre, self, *a) if pre else None
        r = orig(self, *a, **k)
        if post:
            _safe(f"{cls.__name__}.{name}", post, self, ctx, r)
        return r

    wrapped.__wrapped__ = orig
    setattr(cls, name, wrapped)


# ----- teleop process
_tp = {"seq": 0, "last_ts": None, "last_read": None}


def install_teleop():
    from decoupled_wbc.control.policy import teleop_policy as TP
    from decoupled_wbc.control.teleop import teleop_retargeting_ik as TRI
    from decoupled_wbc.control.teleop import teleop_streamer as TS
    from decoupled_wbc.control.teleop.streamers import pico_streamer as PS
    from decoupled_wbc.control.utils import ros_utils as RU

    _open("lat_teleop.csv", TELEOP_COLS)
    mono = time.monotonic

    def pico_post(self, ctx, d):
        t = mono()
        ts = int(d.get("timestamp") or 0)
        _tp["t_read"], _tp["wall_ns"], _tp["pico_ts_ns"] = t, time.time_ns(), ts
        _tp["pico_new"] = ts != _tp["last_ts"]
        _tp["period_ms"] = None if _tp["last_read"] is None else (t - _tp["last_read"]) * 1e3
        _tp["last_ts"], _tp["last_read"] = ts, t

    _wrap(PS.PicoStreamer, "_get_pico_data", None, pico_post)
    _wrap(TS.TeleopStreamer, "get_streamer_data", lambda self: mono(),
          lambda self, t0, r: _tp.__setitem__("streamer_ms", (mono() - t0) * 1e3))
    _wrap(TRI.TeleopRetargetingIK, "get_action", lambda self: mono(),
          lambda self, t0, r: _tp.__setitem__("ik_ms", (mono() - t0) * 1e3))

    def act_pre(self):
        for k in ("t_read", "pico_ts_ns", "wall_ns", "pico_new", "period_ms", "streamer_ms", "ik_ms"):
            _tp[k] = None
        return mono()

    def act_post(self, t0, action):
        _tp["seq"] += 1
        _tp["action_ms"] = (mono() - t0) * 1e3
        _tp["active"] = bool(getattr(self, "is_active", False))
        if isinstance(action, dict):
            action["lat"] = {k: _tp.get(k) for k in ("seq", "t_read", "pico_ts_ns", "wall_ns", "pico_new",
                                                        "period_ms", "streamer_ms", "ik_ms", "action_ms", "active")}

    _wrap(TP.TeleopPolicy, "get_action", act_pre, act_post)

    def pub_pre(self, msg):
        return mono() if isinstance(msg, dict) and "lat" in msg else None

    def pub_post(self, t0, r):
        if t0 is None:
            return
        row = dict(_tp, pub_ms=(mono() - t0) * 1e3)
        _write("lat_teleop.csv", ",".join(_fmt(row.get(c)) for c in TELEOP_COLS))

    _wrap(RU.ROSMsgPublisher, "publish", pub_pre, pub_post)
    print("[dwbc_latency] teleop instrumentation on", flush=True)


# ----- control process
_cp = {"pending": None, "obs_q": None, "arms": None}


def install_control():
    from decoupled_wbc.control.envs.g1 import g1_env as GE
    from decoupled_wbc.control.policy import interpolation_policy as IP
    from decoupled_wbc.control.utils import ros_utils as RU

    _open("lat_control.csv", CONTROL_COLS)
    _open("lat_track.csv", ["t"] + [f"cmd{i}" for i in range(N_ARM)] + [f"meas{i}" for i in range(N_ARM)])
    mono = time.monotonic

    # arrival time of the newest message (ROS executor thread); the class attribute is bound at subscription time
    _wrap(RU.ROSMsgSubscriber, "_callback", None, lambda self, ctx, r: setattr(self, "_lat_arrive", mono()))

    def get_post(self, ctx, msg):
        if isinstance(msg, dict) and isinstance(msg.get("lat"), dict):
            row = dict(msg["lat"])
            row.update(t_pub=msg.get("timestamp"), t_target=msg.get("target_time"),
                       t_arrive=getattr(self, "_lat_arrive", None), t_pick=mono())
            _cp["pending"] = row

    _wrap(RU.ROSMsgSubscriber, "get_msg", None, get_post)

    def goal_post(self, ctx, r):
        if _cp["pending"] is not None and "t_waypoint" not in _cp["pending"]:
            # the speed-capped time of the waypoint just scheduled (self.last_waypoint_time is only target_time)
            _cp["pending"]["t_waypoint"] = float(self.interp.times[-1])

    _wrap(IP.InterpolationPolicy, "set_goal", None, goal_post)

    def obs_post(self, ctx, obs):
        _cp["obs_q"] = obs.get("q") if isinstance(obs, dict) else None

    _wrap(GE.G1Env, "observe", None, obs_post)

    def queue_post(self, ctx, r):
        t = mono()
        row, _cp["pending"] = _cp["pending"], None
        if row is not None:
            row["t_sent"] = t
            _write("lat_control.csv", ",".join(_fmt(row.get(c)) for c in CONTROL_COLS))
        q, cmd = _cp["obs_q"], ctx
        if q is not None and cmd is not None:
            if _cp["arms"] is None:
                _cp["arms"] = list(self.robot_model.get_joint_group_indices("arms"))[:N_ARM]
            a = _cp["arms"]
            _write("lat_track.csv", f"{t:.4f}," + ",".join(f"{cmd[i]:.4f}" for i in a) + ","
                   + ",".join(f"{q[i]:.4f}" for i in a))

    _wrap(GE.G1Env, "queue_action", lambda self, action: action["q"].copy(), queue_post)
    print("[dwbc_latency] control instrumentation on", flush=True)


def install(mode):
    """Called by dwbc_run.py before the upstream entry point runs; never raises."""
    try:
        {"teleop": install_teleop, "control": install_control}[mode]()
    except Exception as e:  # noqa: BLE001
        _off(f"install({mode})", e)


# ------------------------------------------------------------------------------------------------ viewer / report
HOPS = [  # key, label
    ("pico", "[1] PICO采样→teleop读到 (超出前后 60 s 最小值)"),
    ("period", "[2] teleop 周期 (目标 50)"),
    ("streamer", "[3] 读手柄+位姿处理"),
    ("ik", "[4] 手臂 IK"),
    ("other", "[5] teleop 其它"),
    ("ros", "[6-7] 打包+ROS 2 传输"),
    ("wait", "[8] 等控制环取走"),
    ("ctrl", "[9] 控制环计算→lowcmd"),
    ("interp", "[10] 插值到位 (含限速)"),
    ("chain", "工作站内合计 读到→指令到位"),
]


# The overview: PICO motion -> robot at the target, cut into five blocks that add up (p50 each, ms).
BLOCKS = [  # key, label, what it is / what moves it
    ("pico", "PICO→工作站", "Wi‑Fi + XRoboToolkit 服务 = [1]。头显时钟没和工作站同步且会漂移（实测约 2–3 ms/分钟），这是相对前后 60 s 内最快一帧多出来的部分，看波动不看绝对值"),
    ("compute", "工作站计算", "teleop 读到手柄 → 发出目标 = [3]+[4]+[5]+打包，主要是手臂 IK"),
    ("transfer", "传输等待", "ROS 2 传输 + 等 50 Hz 控制环来取 = [6-7]+[8]，后者在 0–20 ms 之间"),
    ("interp", "插值平滑", "控制环取到目标 → 插值轨迹到达它 = [10]。官方设计约 50 ms，用来平滑 20 Hz 的目标；关节限速时更长"),
    ("follow", "机器人跟随", "机器人实测关节落后指令的时间 = [11]（含网线、电机 PD 跟踪），主要由手臂 PD 增益决定"),
]
STALL_MS = 100.0      # a frame "stalls" when PICO→指令到位 is this much above its median
STALL_HOPS = [("pico", "PICO→工作站"), ("streamer", "读手柄"), ("ik", "手臂 IK"), ("other", "teleop 其它"),
              ("ros", "ROS 2 传输"), ("wait", "等控制环"), ("interp", "插值限速")]


def _f(x):
    try:
        return float(x) if x != "" else None
    except ValueError:
        return None


class Tail:
    """Reads the complete lines appended to a CSV since the last call."""

    def __init__(self, path):
        self.path, self.pos, self.cols, self.buf = path, 0, None, ""

    def read(self):
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        data = self.buf + data
        lines = data.split("\n")
        self.buf = lines.pop()
        out = []
        for ln in lines:
            if not ln:
                continue
            parts = ln.split(",")
            if self.cols is None:
                self.cols = parts
                continue
            out.append(dict(zip(self.cols, parts)))
        return out


def control_hops(r, pico_min):
    """r: a lat_control.csv row (strings). Returns {hop: ms} for the hops present. The PICO hop is measured against the
    row's own baseline r["_pb"] (pico_baseline / Live: min offset in a 60 s window) when set, else against pico_min."""
    g = {k: _f(v) for k, v in r.items() if k != "_pb"}
    h = {}
    if g.get("wall_ns") and g.get("pico_ts_ns"):
        h["pico_raw"] = (g["wall_ns"] - g["pico_ts_ns"]) / 1e6
        base = r.get("_pb", pico_min)
        if base is not None:
            h["pico"] = h["pico_raw"] - base
    for k, c in (("period", "period_ms"), ("streamer", "streamer_ms"), ("ik", "ik_ms")):
        if g.get(c) is not None:
            h[k] = g[c]
    if g.get("action_ms") is not None:
        h["other"] = max(0.0, g["action_ms"] - (g.get("streamer_ms") or 0) - (g.get("ik_ms") or 0))

    def d(a, b):
        return (g[b] - g[a]) * 1e3 if g.get(a) is not None and g.get(b) is not None else None

    for k, a, b in (("ros", "t_pub", "t_arrive"), ("wait", "t_arrive", "t_pick"), ("ctrl", "t_pick", "t_sent"),
                    ("interp", "t_pick", "t_waypoint"), ("chain", "t_read", "t_waypoint")):
        v = d(a, b)
        if v is not None:
            h[k] = v
    # the five blocks of the overview (BLOCKS); pico + compute + transfer + interp == pico + chain for every frame
    c = d("t_read", "t_pub")
    if c is not None:
        h["compute"] = c
    if h.get("ros") is not None and h.get("wait") is not None:
        h["transfer"] = h["ros"] + h["wait"]
    return h


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def follow_lag(rows, max_lag=40):
    """Robot follow lag (ms) of the arm joints: for each moving joint the shift of the command that best matches the
    measurement (mean removed: gravity sag is an offset, not a lag). Returns (lag_ms, n_joints, rms_err_deg)."""
    if len(rows) < 60:
        return None, 0, None
    t = [_f(r["t"]) for r in rows]
    dt = (t[-1] - t[0]) / (len(t) - 1)
    lags, errs = [], []
    for j in range(N_ARM):
        c = [_f(r.get(f"cmd{j}")) for r in rows]
        m = [_f(r.get(f"meas{j}")) for r in rows]
        if None in c or None in m:
            continue
        errs.extend((a - b) ** 2 for a, b in zip(c, m))
        if max(c) - min(c) < 0.05:     # < 3 deg of motion: no lag to measure
            continue
        n = len(c)
        best = None
        for L in range(0, min(max_lag, n // 3) + 1):
            cc, mm = c[:n - L], m[L:]
            mc, mmn = sum(cc) / len(cc), sum(mm) / len(mm)
            e = sum((a - mc - (b - mmn)) ** 2 for a, b in zip(cc, mm)) / len(cc)
            if best is None or e < best[0]:
                best = (e, L)
        lags.append(best[1] * dt * 1e3)
    rms = (sum(errs) / len(errs)) ** 0.5 * 57.2958 if errs else None
    return (pct(lags, 0.5) if lags else None), len(lags), rms


def resolve_dir(arg):
    d = arg or os.path.expanduser("~/piper_ws/dwbc_logs/real_latest")
    return os.path.realpath(d)


def pico_offset(r):
    a, b = _f(r.get("wall_ns")), _f(r.get("pico_ts_ns"))
    return (a - b) / 1e6 if a and b else None


def summary(ctrl_rows, track_rows, pico_min, window_s=None, lag_windows=None, lag_values=None):
    """JSON-able summary of control rows (only frames with the arms following) + follow lag. Used by watch / report and
    by g1_hub (state.link.dwbc, dwbc_lat_detail). lag_windows: list of track-row windows to take the median lag over
    (whole runs); lag_values: those windows' lags already computed; else the lag of track_rows itself.

    For the app (easy to read): blocks (five parts of e2e that add up, the largest = worst), stall (how many frames are
    STALL_MS above the usual and which hop causes them). hops keeps every hop's p50 / p90 / max (the details)."""
    rows = [r for r in ctrl_rows if r.get("active") == "1"]
    vals, per = {}, []
    for r in rows:
        h = control_hops(r, pico_min)
        per.append(h)
        for k, v in h.items():
            vals.setdefault(k, []).append(v)
    if lag_windows is not None and lag_values is None:
        lag_values = [follow_lag(w)[0] for w in lag_windows]
    if lag_values is not None:
        lags = [lg for lg in lag_values if lg is not None]
        lag, nj = (pct(lags, 0.5) if lags else None), len(lags)
        rms = follow_lag(track_rows)[2] if len(track_rows or []) >= 60 else None
    else:
        lag, nj, rms = follow_lag(track_rows)
    hops = []
    for k, label in HOPS:
        xs = vals.get(k, [])
        hops.append({"key": k, "label": label, "p50": _r(pct(xs, 0.5)), "p90": _r(pct(xs, 0.9)), "max": _r(pct(xs, 1.0)),
                     "n": len(xs)})
    hops.append({"key": "follow", "label": "[11] 机器人跟随滞后 (实测 vs 指令)", "p50": _r(lag), "p90": None, "max": None,
                 "n": nj})
    p50 = {h["key"]: h["p50"] for h in hops}
    # overview: five blocks, p50 each; e2e = their sum
    blocks = []
    for k, label, hint in BLOCKS:
        v = _r(lag) if k == "follow" else _r(pct(vals.get(k, []), 0.5))
        blocks.append({"key": k, "label": label, "ms": v, "hint": hint})
    have = [b for b in blocks if b["ms"] is not None]
    e2e = _r(sum(b["ms"] for b in have)) if any(b["key"] == "compute" for b in have) else None
    for b in blocks:
        b["pct"] = _r(100.0 * b["ms"] / e2e, 0) if e2e and b["ms"] is not None else None
    wb = max(have, key=lambda b: b["ms"]) if e2e else None
    stall = _stall(per)
    # the biggest single hop inside the chain (what to look at first)
    parts = [h for h in hops if h["key"] in ("pico", "streamer", "ik", "other", "ros", "wait", "ctrl", "interp", "follow")
             and h["p50"] is not None]
    worst = max(parts, key=lambda h: h["p50"])["key"] if parts else None
    raw = vals.get("pico_raw", [])
    new = sum(1 for r in rows if r.get("pico_new") == "1")
    return {"window_s": window_s, "frames": len(rows), "following": bool(rows), "hops": hops, "e2e_ms": e2e,
            "chain_ms": p50.get("chain"), "follow_lag_ms": _r(lag), "follow_rms_deg": _r(rms, 2),
            "follow_n": nj, "pico_clock_ms": _r(pct(raw, 0.5)),
            "pico_repeat_pct": _r(100.0 * (1 - new / len(rows))) if rows else None, "bottleneck": worst,
            "blocks": blocks, "worst": wb["key"] if wb else None, "worst_label": wb["label"] if wb else None,
            "worst_ms": wb["ms"] if wb else None, "worst_pct": wb["pct"] if wb else None, "stall": stall}


def _stall(per):
    """Frames whose PICO→指令到位 (pico + chain) is STALL_MS (and 2x) above its median, and the hop behind them: the one
    furthest above its own median in each such frame."""
    tot = [(h.get("pico") or 0) + h["chain"] for h in per if h.get("chain") is not None]
    if not tot:
        return None
    base = pct(tot, 0.5)
    med = {k: pct([h[k] for h in per if h.get(k) is not None], 0.5) for k, _ in STALL_HOPS}
    n1 = n2 = 0
    worst = 0.0
    cause = {}
    for h in per:
        if h.get("chain") is None:
            continue
        ex = (h.get("pico") or 0) + h["chain"] - base
        worst = max(worst, ex)
        if ex < STALL_MS:
            continue
        n1 += 1
        n2 += ex >= 2 * STALL_MS
        k = max((k for k, _ in STALL_HOPS if h.get(k) is not None and med[k] is not None),
                key=lambda k: h[k] - med[k], default=None)
        if k:
            cause[k] = cause.get(k, 0) + 1
    ck = max(cause, key=cause.get) if cause else None
    names = dict(STALL_HOPS)
    return {"over_ms": STALL_MS, "frames": len(tot), "n": n1, "pct": _r(100.0 * n1 / len(tot)),
            "pct_2x": _r(100.0 * n2 / len(tot)), "worst_extra_ms": _r(worst, 0),
            "cause": ck, "cause_label": names.get(ck), "cause_pct": _r(100.0 * cause[ck] / n1, 0) if ck else None,
            "causes": [{"key": k, "label": names[k], "n": c} for k, c in sorted(cause.items(), key=lambda kv: -kv[1])]}


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


PICO_BASE_S = 60.0   # the PICO hop is measured against the fastest frame within this window (clock drift, 2026-11-24)


def pico_baseline(rows, window_s=PICO_BASE_S):
    """Sets r["_pb"] on every row with a PICO timestamp: the smallest workstation - PICO offset within +-window_s/2 of
    it (by t_read). The headset clock is not synchronised and drifts against the workstation (2.7 ms/min on 2026-11-23,
    75 ms over a 28 min run), so one minimum for a whole run counts the drift as Wi-Fi delay. O(n), monotonic deque."""
    pts = []
    for r in rows:
        o, t = pico_offset(r), _f(r.get("t_read"))
        if o is not None and t is not None:
            pts.append((t, o, r))
    half = window_s / 2
    q, j = collections.deque(), 0          # indexes into pts, offsets increasing; window [t - half, t + half]
    for t, _, r in pts:
        while j < len(pts) and pts[j][0] <= t + half:
            while q and pts[q[-1]][1] >= pts[j][1]:
                q.pop()
            q.append(j)
            j += 1
        while q and pts[q[0]][0] < t - half:
            q.popleft()
        r["_pb"] = pts[q[0]][1]


def read_run(d):
    ctrl = Tail(os.path.join(d, "lat_control.csv")).read()
    track = Tail(os.path.join(d, "lat_track.csv")).read()
    offs = [o for o in (pico_offset(r) for r in ctrl) if o is not None]
    pico_baseline(ctrl)
    return ctrl, track, (min(offs) if offs else None)


def run_summary(d, series=True, bin_s=1.0):
    """Whole run: the summary plus (series=True) a 1 s time series of the main hops (p50 per bin, following frames)."""
    ctrl, track, pico_min = read_run(d)
    wins = [track[i:i + 250] for i in range(0, len(track), 250)]
    wlag = [follow_lag(w)[0] for w in wins]          # 5 s windows, computed once (the slow part)
    wt = [_f(w[0].get("t")) for w in wins]
    out = summary(ctrl, track, pico_min, None, lag_values=wlag)
    out["segments"] = _segments(ctrl, track, pico_min, wt, wlag)
    out["run"] = os.path.basename(d)
    t0s = [_f(r.get("t_pick")) for r in ctrl[:1]] + [_f(r.get("t")) for r in track[:1]]
    t0 = min([t for t in t0s if t is not None], default=None)
    out["start_mono"] = t0
    out["duration_s"] = _r(max([_f(r.get("t")) or 0 for r in track[-1:]] + [_f(r.get("t_pick")) or 0 for r in ctrl[-1:]])
                           - t0) if t0 is not None else None
    if series and t0 is not None:
        keys = ("e2e", "chain", "ik", "period", "interp", "pico", "follow")
        bins = {}
        for r in ctrl:
            if r.get("active") != "1":
                continue
            b = int(((_f(r.get("t_pick")) or t0) - t0) / bin_s)
            h = control_hops(r, pico_min)
            dst = bins.setdefault(b, {k: [] for k in keys})
            for k in ("chain", "ik", "period", "interp", "pico"):
                if h.get(k) is not None:
                    dst[k].append(h[k])
        lag_by_bin = {}
        for t, lg in zip(wt, wlag):                       # follow lag on 5 s windows, spread over its bins
            if lg is not None:
                b = int(((t or t0) - t0) / bin_s)
                for j in range(int(round(5 / bin_s))):
                    lag_by_bin[b + j] = lg
        s = {k: [] for k in ("t",) + keys}
        for b in sorted(bins):
            v = {k: pct(bins[b][k], 0.5) for k in ("chain", "ik", "period", "interp", "pico")}
            v["follow"] = lag_by_bin.get(b)
            v["e2e"] = (v["chain"] + (v["pico"] or 0) + (v["follow"] or 0)) if v["chain"] is not None else None
            s["t"].append(round(b * bin_s, 1))
            for k in keys:
                s[k].append(_r(v[k]))
        out["series"] = s
    return out


def _segments(ctrl, track, pico_min, wt, wlag):
    """One summary per A+X following stretch (active 0 -> 1 -> 0), with its start / end (unix, from the rows' wall
    clock) and seconds since the run start. Follow lag: the 5 s windows that start inside the stretch."""
    spans, cur = [], None
    for r in ctrl:
        if r.get("active") == "1":
            cur = cur or []
            cur.append(r)
        elif cur:
            spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    t0 = _f(ctrl[0].get("t_read")) if ctrl else None
    out = []
    for i, rs in enumerate(spans, 1):
        a, b = _f(rs[0].get("t_read")), _f(rs[-1].get("t_read"))
        if a is None or b is None:
            continue
        tr = [r for r in track if a <= (_f(r.get("t")) or 0) <= b]
        lv = [lg for t, lg in zip(wt, wlag) if t is not None and a <= t <= b - 2.0]
        s = summary(rs, tr, pico_min, None, lag_values=lv if lv else None)
        w0, w1 = _f(rs[0].get("wall_ns")), _f(rs[-1].get("wall_ns"))
        s.update(segment=i, start=_r(w0 / 1e9, 3) if w0 else None, end=_r(w1 / 1e9, 3) if w1 else None,
                 start_s=_r(a - t0) if t0 is not None else None, duration_s=_r(b - a))
        out.append(s)
    return out


def list_runs(logs_dir, limit=50):
    """Runs (real_*) that have latency files, newest first: name, start (unix), size."""
    out = []
    try:
        import re
        names = sorted((n for n in os.listdir(logs_dir) if re.fullmatch(r"real_\d{8}_\d{6}", n)
                        and not os.path.islink(os.path.join(logs_dir, n))), reverse=True)
    except OSError:
        return out
    for n in names:
        p = os.path.join(logs_dir, n, "lat_control.csv")
        if os.path.isfile(p):
            st = os.stat(p)
            try:
                start = time.mktime(time.strptime(n[5:20], "%Y%m%d_%H%M%S"))
            except ValueError:
                start = None
            out.append({"run": n, "start": start, "updated": st.st_mtime, "bytes": st.st_size})
            if len(out) >= limit:
                break
    return out


def _pad(s, w=40):
    """left-justify to w terminal columns (CJK characters are 2 columns wide)"""
    import unicodedata
    n = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(1, w - n)


def render(title, s):
    out = [title, ""]
    if s.get("e2e_ms") is not None:
        out.append(f"估算端到端 {s['e2e_ms']:.0f} ms（中位数）= " + " + ".join(
            f"{b['label']} {b['ms']:.0f}" for b in s["blocks"] if b["ms"] is not None))
        out.append(f"延时最大的一段：{s['worst_label']} {s['worst_ms']:.0f} ms，占 {s['worst_pct']:.0f}%")
        st = s.get("stall")
        if st:
            out.append((f"最大卡顿来源：{st['cause_label']}（占卡顿帧 {st['cause_pct']:.0f}%）；" if st.get("cause") else "")
                       + f"卡顿帧 {st['pct']:.1f}%：比平时慢 {st['over_ms']:.0f} ms 以上（{st['pct_2x']:.1f}% 慢 "
                       f"{2 * st['over_ms']:.0f} ms 以上，最多慢 {st['worst_extra_ms']:.0f} ms）")
        out.append("")
    out.append(_pad("明细 环节") + f"{'p50':>9}{'p90':>9}{'max':>9}   (ms)")
    for h in s["hops"]:
        if h["key"] == "follow":
            continue
        if h["p50"] is None:
            out.append(_pad(h["label"]) + f"{'—':>9}")
        else:
            out.append(_pad(h["label"]) + "".join(f"{h[k]:9.1f}" for k in ("p50", "p90", "max")))
    lag = s["follow_lag_ms"]
    out.append(_pad("[11] 机器人跟随滞后 (实测 vs 指令)")
               + (f"{lag:9.0f}   (跟随误差 RMS {s['follow_rms_deg'] or 0:.1f}°)" if lag is not None else
                  f"{'—':>9}   (手臂没在动或数据不足)"))
    if s["frames"]:
        out.append("")
        out.append(f"帧数 {s['frames']}，其中 PICO 数据没更新的重复帧 {s['pico_repeat_pct']:.0f}%"
                   + (f"；PICO 时钟差 p50 {s['pico_clock_ms']:.0f} ms（两边时钟同步时即 Wi‑Fi 绝对延迟）"
                      if s["pico_clock_ms"] is not None else ""))
    for g in s.get("segments") or []:
        if g["segment"] == 1:
            out += ["", "按 A+X 分段："]
        st = g.get("stall") or {}
        out.append(f"  第 {g['segment']} 段 {time.strftime('%H:%M:%S', time.localtime(g['start'])) if g.get('start') else '?'}"
                   f" {g['duration_s']:>5.0f} s  端到端 {g['e2e_ms'] or 0:>4.0f}  延时最大 {g['worst_label'] or '–'} "
                   f"{g['worst_ms'] or 0:.0f}  卡顿帧 {st.get('pct') or 0:.0f}%"
                   + (f"（最大卡顿来源 {st['cause_label']}）" if st.get("cause") else ""))
    return "\n".join(out)


class Live:
    """Incremental reader of a run's latency files for a rolling window (watch, g1_hub)."""

    def __init__(self, d, window=5.0):
        self.d, self.window = d, window
        self.tc, self.tt = Tail(os.path.join(d, "lat_control.csv")), Tail(os.path.join(d, "lat_track.csv"))
        self.ctrl, self.track, self.pico_min, self.last_t = [], [], None, None
        self._pq = collections.deque()    # (t_read, offset) of the last PICO_BASE_S s, offsets increasing

    def update(self):
        new = self.tc.read()
        for r in new:                     # trailing 60 s minimum as each row's PICO baseline (clock drift)
            v, t = pico_offset(r), _f(r.get("t_read"))
            if v is not None:
                self.pico_min = v if self.pico_min is None else min(self.pico_min, v)
            if v is not None and t is not None:
                while self._pq and self._pq[-1][1] >= v:
                    self._pq.pop()
                self._pq.append((t, v))
                while self._pq[0][0] < t - PICO_BASE_S:
                    self._pq.popleft()
                r["_pb"] = self._pq[0][1]
        self.ctrl.extend(new)
        self.track.extend(self.tt.read())
        ts = [_f(r.get("t_pick")) for r in self.ctrl[-1:]] + [_f(r.get("t")) for r in self.track[-1:]]
        ts = [t for t in ts if t is not None]
        if ts:
            self.last_t = max(ts)
            now = self.last_t
            self.ctrl = [r for r in self.ctrl if (_f(r.get("t_pick")) or 0) > now - self.window]
            self.track = [r for r in self.track if (_f(r.get("t")) or 0) > now - self.window]
        s = summary(self.ctrl, self.track, self.pico_min, self.window)
        s["run"] = os.path.basename(self.d)
        # newest row's age on the shared CLOCK_MONOTONIC (same machine as the loops)
        s["age_ms"] = _r((time.monotonic() - self.last_t) * 1e3, 0) if self.last_t is not None else None
        return s


def cmd_watch(d, window=5.0):
    live = Live(d, window)
    while True:
        s = live.update()
        title = (f"G1 DWBC 控制链延迟  {os.path.basename(d)}  最近 {window:.0f} s（只算手臂跟随中的帧）  "
                 f"{time.strftime('%H:%M:%S')}")
        body = render(title, s)
        if not s["frames"]:
            body += "\n\n（还没有手臂跟随中的帧：等 A+X 开始跟随，或链路没在跑）"
        sys.stdout.write("\033[H\033[2J" + body + "\n")
        sys.stdout.flush()
        time.sleep(1.0)


def cmd_report(d):
    s = run_summary(d, series=False)
    txt = render(f"G1 DWBC 控制链延迟  {os.path.basename(d)}  整场（只算手臂跟随中的帧；[11] 是各 5 s 窗口的中位数）", s)
    print(txt)
    with open(os.path.join(d, "lat_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("watch", "report"):
        sys.exit(__doc__)
    try:
        (cmd_watch if sys.argv[1] == "watch" else cmd_report)(resolve_dir(sys.argv[2] if len(sys.argv) > 2 else None))
    except KeyboardInterrupt:
        pass
