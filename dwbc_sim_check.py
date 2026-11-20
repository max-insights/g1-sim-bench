"""End-to-end check of the decoupled-WBC chain in MuJoCo, with a scripted operator instead of the PICO.

Needs the control loop running in sim with the ROS key dispatcher (`~/piper_ws/dwbc/dwbc.sh sim`). This script plays
the teleop loop's part: it runs the same TeleopRetargetingIK in-process on scripted wrist / finger targets and
publishes the same goal message (target_upper_body_pose + navigate_cmd + base_height_command, 20 Hz,
target_time = now + 0.05 s) on ControlPolicy/upper_body_pose. Keys go over /keyboard_input.

Phases: hold (elastic band on) -> ']' lower-body policy on -> '9' band off -> stand -> arm circles -> grippers ->
walk forward -> turn -> crouch -> stand. Reads G1Env/env_state_act (q, wrist FK, sim odometry) and prints a JSON
summary; with --out DIR it also writes samples.npz and check.png.

    source ~/piper_ws/dwbc/dwbc_env.sh; python ~/piper_ws/dwbc/dwbc_sim_check.py --out ~/piper_ws/dwbc_logs/simcheck
"""
import argparse
import json
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    KEYBOARD_INPUT_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
ap.add_argument("--rate", type=float, default=20.0, help="goal publish rate (teleop loop default 20 Hz)")
args = ap.parse_args()

ros = ROSManager(node_name="dwbc_sim_check")
node = ros.node
goal_pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
key_pub = node.create_publisher(String, KEYBOARD_INPUT_TOPIC, 10)
state_sub = ROSMsgSubscriber(STATE_TOPIC_NAME)

robot_model = instantiate_g1_robot_model(waist_location="lower_body")
lh_solver, rh_solver = instantiate_g1_hand_ik_solver()
ik = TeleopRetargetingIK(robot_model=robot_model, left_hand_ik_solver=lh_solver, right_hand_ik_solver=rh_solver,
                         enable_visualization=False, body_active_joint_groups=["upper_body"])
LW = robot_model.supplemental_info.hand_frame_names["left"]
RW = robot_model.supplemental_info.hand_frame_names["right"]


def group(name):
    try:
        return np.asarray(robot_model.get_joint_group_indices(name))
    except Exception:
        return np.zeros(0, dtype=int)


IDX_ARMS, IDX_LH, IDX_RH = group("arms"), group("left_hand"), group("right_hand")


def key(k):
    m = String()
    m.data = k
    key_pub.publish(m)
    print(f"  key {k!r}", flush=True)


def fingers(closed):
    """25 fingertip transforms like the dummy / PICO streamers: thumb (4) far away = open, index (9) on it = closed."""
    f = np.zeros((25, 4, 4))
    f[4, 0, 3] = 1.0
    if closed:
        f[9, 0, 3] = 1.0
    return {"position": f}


def mat(pos, quat_wxyz):
    m = np.eye(4)
    m[:3, :3] = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    m[:3, 3] = pos
    return m


# ---------------------------------------------------------------- state sampling (background, ~50 Hz)
S = {k: [] for k in ("t", "base", "q", "wrist", "action")}
lock = threading.Lock()
stop = threading.Event()


def sampler():
    while not stop.is_set():
        m = state_sub.get_msg()
        if m is not None:
            with lock:
                S["t"].append(time.monotonic())
                S["base"].append(np.asarray(m["floating_base_pose"], dtype=float))
                S["q"].append(np.asarray(m["q"], dtype=float))
                S["wrist"].append(np.asarray(m["wrist_pose"], dtype=float))
                S["action"].append(np.asarray(m.get("action", np.full(len(m["q"]), np.nan)), dtype=float))
        time.sleep(0.004)


threading.Thread(target=sampler, daemon=True).start()
t0 = time.monotonic()
while not S["t"]:
    if time.monotonic() - t0 > 20:
        raise SystemExit("no G1Env/env_state_act within 20 s: is `dwbc.sh sim` running (same ROS_DOMAIN_ID)?")
    time.sleep(0.1)
with lock:
    w0 = S["wrist"][-1].copy()
L0, R0 = mat(w0[0:3], w0[3:7]), mat(w0[7:10], w0[10:14])
print(f"state OK; wrist L {np.round(w0[0:3], 3)} R {np.round(w0[7:10], 3)}", flush=True)

# ---------------------------------------------------------------- scripted operator
phase_log = []  # (name, t_start, t_end)
targets = []  # (t, left_pos(3), right_pos(3)) commanded wrist positions


def run_phase(name, dur, nav=(0.0, 0.0, 0.0), height=DEFAULT_BASE_HEIGHT, wrists=None, grip=(False, False),
              first=False):
    """Publish goals at args.rate for dur seconds. wrists(tau) -> (L 4x4, R 4x4); default: initial wrist poses."""
    print(f"[{name}] {dur:.0f} s nav={nav} height={height:.2f}", flush=True)
    ts = time.monotonic()
    while True:
        now = time.monotonic()
        tau = now - ts
        if tau > dur:
            break
        Lw, Rw = wrists(tau) if wrists else (L0, R0)
        ik.set_goal({"body_data": {LW: Lw, RW: Rw}, "left_hand_data": fingers(grip[0]),
                     "right_hand_data": fingers(grip[1])})
        upper = ik.get_action()
        goal = {"target_upper_body_pose": upper, "navigate_cmd": list(nav), "base_height_command": height,
                "timestamp": now, "target_time": now + (2.0 if first else 1.0 / args.rate)}
        goal_pub.publish(goal)
        targets.append((now, Lw[:3, 3].copy(), Rw[:3, 3].copy()))
        if first:
            time.sleep(2.0)
            first = False
        time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - now)))
    phase_log.append((name, ts, time.monotonic()))


def circles(tau):
    """Right wrist: 6 cm circle in the y-z plane at 0.4 Hz; left wrist: 6 cm forward / back at 0.3 Hz."""
    Rw, Lw = R0.copy(), L0.copy()
    a = 2 * np.pi * 0.4 * tau
    Rw[1, 3] += 0.06 * np.sin(a)
    Rw[2, 3] += 0.06 * (1 - np.cos(a))
    Lw[0, 3] += 0.06 * np.sin(2 * np.pi * 0.3 * tau)
    return Lw, Rw


run_phase("hold (band on)", 3, first=True)
key("]")
run_phase("policy on, band still on", 2)
key("9")
run_phase("stand", 6)
run_phase("arm circles", 12, wrists=circles)
run_phase("settle", 2)
run_phase("grippers closed", 3, grip=(True, True))
run_phase("grippers open", 3)
run_phase("walk forward 0.3 m/s", 5, nav=(0.3, 0.0, 0.0))
run_phase("stop", 3)
run_phase("turn 0.5 rad/s", 4, nav=(0.0, 0.0, 0.5))
run_phase("stop 2", 3)
run_phase("crouch 0.64 m", 4, height=0.64)
run_phase("stand up", 4)
stop.set()
time.sleep(0.1)

# ---------------------------------------------------------------- analysis
with lock:
    t = np.asarray(S["t"]) - t0
    base = np.asarray(S["base"])
    q = np.asarray(S["q"])
    wr = np.asarray(S["wrist"])
    act = np.asarray(S["action"])
ph = {n: (a - t0, b - t0) for n, a, b in phase_log}
pos, quat = base[:, 0:3], base[:, 3:7]
rot = R.from_quat(quat, scalar_first=True)
up = rot.apply([0, 0, 1])
tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])


def sel(name, skip=0.0):
    a, b = ph[name]
    return (t >= a + skip) & (t <= b)


after_release = t >= ph["stand"][0] + 0.5
res = {"samples": int(len(t)), "state_rate_hz": round(float(len(t) / (t[-1] - t[0])), 1)}
res["balance"] = {"z_min_m": round(float(pos[after_release, 2].min()), 3),
                  "z_stand_mean_m": round(float(pos[sel("stand", 2), 2].mean()), 3),
                  "tilt_max_deg": round(float(tilt[after_release].max()), 1),
                  "fell": bool(pos[after_release, 2].min() < 0.45 or tilt[after_release].max() > 35)}

# arm tracking: measured wrist FK vs commanded wrist position (best lag 0..0.6 s)
tt = np.array([x[0] for x in targets]) - t0
tl = np.array([x[1] for x in targets])
trr = np.array([x[2] for x in targets])
k = sel("arm circles", 2.0)


def track(meas, tgt):
    """Best lag 0..0.6 s; splits the error into a static offset (gravity sag: the upstream loop sends no
    feed-forward torque, enable_gravity_compensation is only stored) and the dynamic residual around it."""
    best = None
    for lag in np.arange(0.0, 0.61, 0.02):
        ref = np.stack([np.interp(t[k] - lag, tt, tgt[:, i]) for i in range(3)], 1)
        ev = meas[k] - ref
        off = ev.mean(0)
        dyn = np.linalg.norm(ev - off, axis=1)
        rms_dyn = float(np.sqrt(np.mean(dyn ** 2)))
        if best is None or rms_dyn < best[1]:
            e = np.linalg.norm(ev, axis=1)
            best = (lag, rms_dyn, float(np.sqrt(np.mean(e ** 2))), off)
    return {"lag_s": round(best[0], 2), "dynamic_rms_cm": round(100 * best[1], 2),
            "total_rms_cm": round(100 * best[2], 2), "static_offset_cm": np.round(100 * best[3], 1).tolist()}


res["arm_tracking"] = {"left": track(wr[:, 0:3], tl), "right": track(wr[:, 7:10], trr)}
if len(IDX_ARMS):
    e = q[k][:, IDX_ARMS] - act[k][:, IDX_ARMS]
    res["arm_tracking"]["joint_cmd_vs_meas_rms_rad"] = round(float(np.sqrt(np.nanmean(e ** 2))), 3)
if len(IDX_LH):
    c, o = sel("grippers closed", 1.5), sel("grippers open", 1.5)
    res["grippers"] = {s: round(float(np.abs(q[c][:, idx].mean(0) - q[o][:, idx].mean(0)).max()), 2)
                       for s, idx in (("left_max_joint_change_rad", IDX_LH), ("right_max_joint_change_rad", IDX_RH))}

a, b = ph["walk forward 0.3 m/s"]
i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b) - 1
fwd = R.from_euler("Z", yaw[i0]).inv().apply(pos[i1] - pos[i0])
res["walk"] = {"forward_m": round(float(fwd[0]), 2), "lateral_m": round(float(fwd[1]), 2),
               "speed_mps": round(float(fwd[0] / (t[i1] - t[i0])), 2)}
s2 = sel("stop", 1.5)
v = np.linalg.norm(np.diff(pos[s2, :2], axis=0), axis=1) / np.diff(t[s2])
res["walk"]["speed_after_stop_mps"] = round(float(np.median(v)), 3)
a, b = ph["turn 0.5 rad/s"]
i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b) - 1
res["turn"] = {"yaw_change_deg": round(float(np.degrees(yaw[i1] - yaw[i0])), 1),
               "rate_dps": round(float(np.degrees(yaw[i1] - yaw[i0]) / (t[i1] - t[i0])), 1)}
res["crouch"] = {"z_stand_m": res["balance"]["z_stand_mean_m"],
                 "z_crouch_m": round(float(pos[sel("crouch 0.64 m", 2.5), 2].mean()), 3),
                 "z_back_m": round(float(pos[sel("stand up", 2.5), 2].mean()), 3)}

checks = {
    "stands without falling": not res["balance"]["fell"],
    "arm dynamic tracking rms < 2 cm": max(res["arm_tracking"]["left"]["dynamic_rms_cm"],
                                           res["arm_tracking"]["right"]["dynamic_rms_cm"]) < 2.0,
    "arm static sag < 5 cm": max(np.linalg.norm(res["arm_tracking"][s]["static_offset_cm"])
                                 for s in ("left", "right")) < 5.0,
    "walks forward > 0.6 m in 5 s": res["walk"]["forward_m"] > 0.6,
    "stops (< 0.1 m/s)": res["walk"]["speed_after_stop_mps"] < 0.1,
    "turns > 45 deg in 4 s": res["turn"]["yaw_change_deg"] > 45,
    "crouches > 4 cm": res["crouch"]["z_stand_m"] - res["crouch"]["z_crouch_m"] > 0.04,
}
if "grippers" in res:
    checks["grippers move > 0.3 rad"] = min(res["grippers"].values()) > 0.3
checks = {k: bool(v) for k, v in checks.items()}
res["checks"] = checks
res["pass"] = all(checks.values())
res["phases_s"] = {n: [round(a, 1), round(b, 1)] for n, (a, b) in ph.items()}
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)

if args.out:
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "samples.npz"), t=t, base=base, q=q, wrist=wr, action=act,
                        target_t=tt, target_left=tl, target_right=trr)
    json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(t, pos[:, 2]); ax[0].set_ylabel("base z (m)")
    ax[1].plot(t, tilt); ax[1].set_ylabel("tilt (deg)")
    ax[2].plot(t, pos[:, 0], label="x"); ax[2].plot(t, pos[:, 1], label="y")
    ax[2].plot(t, np.degrees(yaw) / 100, label="yaw/100 (deg)"); ax[2].legend(); ax[2].set_ylabel("odometry")
    ax[3].plot(tt, trr[:, 2], "--", label="R wrist z target"); ax[3].plot(t, wr[:, 9], label="R wrist z")
    ax[3].plot(tt, tl[:, 0], "--", label="L wrist x target"); ax[3].plot(t, wr[:, 0], label="L wrist x")
    ax[3].legend(); ax[3].set_ylabel("m"); ax[3].set_xlabel("s")
    for a_ in ax:
        for n, (a, b) in ph.items():
            a_.axvline(a, color="0.85", lw=0.8)
    for n, (a, b) in ph.items():
        ax[0].text(a, ax[0].get_ylim()[1], n, fontsize=7, rotation=90, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "check.png"), dpi=110)
    print("wrote", args.out, flush=True)
ros.shutdown()
os._exit(0 if res["pass"] else 1)
