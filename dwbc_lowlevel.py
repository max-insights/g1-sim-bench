"""Low-level helpers for the decoupled-WBC real chain (Unitree DDS on the robot cable).

    python dwbc_lowlevel.py check   [--iface enp13s0] [--json OUT]   read-only preflight (publishes nothing)
    python dwbc_lowlevel.py release                                  MotionSwitcher ReleaseMode until no mode is active
    python dwbc_lowlevel.py ramp    [--seconds 3]                    all 29 joints -> WBC default pose (MOVES THE ROBOT)
    python dwbc_lowlevel.py damp    [--seconds 1.5]                  kp 0 / kd 8 on all 29 motors (SONIC's e-stop)

The ramp uses the WBC yaml MOTOR_KP/KD (the gains the control loop uses next), so the hand-over to the control loop
(which first holds the current leg angles) is continuous. Default pose: gear_wbc default_angles for legs + waist and
the robot model's initial upper-body pose for the arms (where the control loop's own 2 s arm ramp goes).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, IMUState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

REPO = os.environ.get("DWBC_REPO", os.path.expanduser("~/piper_ws/GR00T-WholeBodyControl"))
WBC_YAML = f"{REPO}/decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml"
GEAR_YAML = f"{REPO}/decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"
N = 29
NAMES = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll", "R_hip_pitch",
         "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll", "waist_yaw", "waist_roll",
         "waist_pitch", "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist_roll",
         "L_wrist_pitch", "L_wrist_yaw", "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
         "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"]


class Rate:
    def __init__(self, topic, typ):
        self.n, self.last, self.t_last = 0, None, 0.0
        self.sub = ChannelSubscriber(topic, typ)
        self.sub.Init(self._cb, 10)

    def _cb(self, m):
        self.n += 1
        self.last = m
        self.t_last = time.monotonic()


def wbc_config():
    cfg = yaml.safe_load(open(WBC_YAML))
    gear = yaml.safe_load(open(GEAR_YAML))
    return cfg, gear


def default_pose():
    """29 motor angles: gear_wbc default_angles (legs 12 + waist 3) + robot model initial upper-body arms."""
    _, gear = wbc_config()
    q = np.zeros(N)
    q[:15] = gear["default_angles"][:15]
    try:
        sys.path.insert(0, REPO)
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

        rm = instantiate_g1_robot_model(waist_location="lower_body")
        full = np.zeros(rm.num_dofs)
        ub = np.asarray(rm.get_joint_group_indices("upper_body"))
        full[ub] = rm.get_initial_upper_body_pose()
        body = rm.get_body_actuated_joints(full)  # actuator order, 29
        q[15:] = np.asarray(body)[15:]
    except Exception as e:  # arms stay at 0 (straight down) if the model cannot be built
        print(f"!! robot model unavailable ({e}); arms ramp to 0", flush=True)
    return q


def wait_state(ls, timeout=3.0):
    t0 = time.monotonic()
    while ls.last is None:
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(0.01)
    return ls.last


def make_cmd(mode_machine):
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = mode_machine
    for i in range(N):
        cmd.motor_cmd[i].mode = 1
    return cmd


def check(a):
    ls = Rate("rt/lowstate", LowState_)
    imu2 = Rate("rt/secondary_imu", IMUState_)
    lc = Rate("rt/lowcmd", LowCmd_)
    d1 = {s: Rate(f"rt/dex1/{s}/state", MotorStates_) for s in ("left", "right")}
    d3 = {s: Rate(f"rt/dex3/{s}/state", HandState_) for s in ("left", "right")}
    time.sleep(a.seconds)
    T = a.seconds
    out = {"lowstate_hz": round(ls.n / T), "secondary_imu_hz": round(imu2.n / T), "lowcmd_hz": round(lc.n / T),
           "dex1_hz": {s: round(r.n / T) for s, r in d1.items()}, "dex3_state_hz": {s: round(r.n / T) for s, r in d3.items()}}
    if ls.last is not None:
        m = ls.last
        out["mode_machine"] = int(m.mode_machine)
        cas = [int(m.motor_state[i].temperature[0]) for i in range(N)]
        win = [int(m.motor_state[i].temperature[1]) for i in range(N)]
        out["hottest_casing"] = [NAMES[int(np.argmax(cas))], max(cas)]
        out["hottest_winding"] = [NAMES[int(np.argmax(win))], max(win)]
        out["q"] = [round(float(m.motor_state[i].q), 3) for i in range(N)]
        rpy = np.degrees(np.asarray(m.imu_state.rpy))
        out["pelvis_rpy_deg"] = [round(float(x), 1) for x in rpy]
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        msc = MotionSwitcherClient()
        msc.SetTimeout(3.0)
        msc.Init()
        code, res = msc.CheckMode()
        out["motion_mode"] = (res or {}).get("name", "?") if code == 0 else f"rpc error {code}"
    except Exception as e:
        out["motion_mode"] = f"unavailable ({e})"
    problems = []
    if out["lowstate_hz"] < 200:
        problems.append("no / slow rt/lowstate")
    if out["secondary_imu_hz"] < 50:
        problems.append("no rt/secondary_imu (the WBC body state needs the torso IMU)")
    if out.get("mode_machine") not in (None, 5) and os.environ.get("DWBC_DRY") != "1":  # the fake robot sends 0
        problems.append(f"mode_machine {out.get('mode_machine')} != 5 (the WBC yaml value)")
    if out.get("hottest_casing", ["", 0])[1] >= 70 or out.get("hottest_winding", ["", 0])[1] >= 95:
        problems.append(f"hot motor: casing {out.get('hottest_casing')} winding {out.get('hottest_winding')}")
    out["problems"] = problems
    out["ok"] = not problems
    print(json.dumps({k: v for k, v in out.items() if k != "q"}, ensure_ascii=False), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    return 0 if out["ok"] else 2


def release(a):
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    for i in range(10):
        code, res = msc.CheckMode()
        name = (res or {}).get("name", "") if code == 0 else None
        print(f"CheckMode code={code} mode={name!r}", flush=True)
        if code == 0 and not name:
            print("no motion mode active: low-level control is free", flush=True)
            return 0
        code, _ = msc.ReleaseMode()
        print(f"ReleaseMode code={code}", flush=True)
        time.sleep(1.0)
    print("!! mode still active after 10 tries", flush=True)
    return 1


def ramp(a):
    cfg, _ = wbc_config()
    kp, kd = np.asarray(cfg["MOTOR_KP"], float), np.asarray(cfg["MOTOR_KD"], float)
    if a.kp_scale != 1.0:
        kp = kp * a.kp_scale
    ls = Rate("rt/lowstate", LowState_)
    m = wait_state(ls)
    if m is None:
        print("!! no rt/lowstate: not ramping", flush=True)
        return 1
    q0 = np.array([m.motor_state[i].q for i in range(N)])
    q1 = default_pose()
    print("ramp max |dq| %.2f rad (%s) over %.1f s" % (np.abs(q1 - q0).max(), NAMES[int(np.argmax(np.abs(q1 - q0)))],
                                                      a.seconds), flush=True)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = make_cmd(int(m.mode_machine))
    dt = 1.0 / a.hz
    steps = int(a.seconds * a.hz)
    t_next = time.monotonic()
    for k in range(steps + int(0.5 * a.hz)):  # + 0.5 s hold at the end
        s = min(1.0, (k + 1) / steps)
        s = s * s * (3 - 2 * s)  # smoothstep: zero start / end velocity
        q = q0 + s * (q1 - q0)
        for i in range(N):
            c = cmd.motor_cmd[i]
            c.q, c.dq, c.tau, c.kp, c.kd = float(q[i]), 0.0, 0.0, float(kp[i]), float(kd[i])
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t_next += dt
        time.sleep(max(0.0, t_next - time.monotonic()))
    m = ls.last
    err = np.array([m.motor_state[i].q for i in range(N)]) - q1
    print("ramp done: max |q - target| %.3f rad (%s)" % (np.abs(err).max(), NAMES[int(np.argmax(np.abs(err)))]),
          flush=True)
    return 0


def damp(a):
    ls = Rate("rt/lowstate", LowState_)
    m = wait_state(ls, 1.0)
    mm = int(m.mode_machine) if m is not None else 5
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = make_cmd(mm)
    for i in range(N):
        c = cmd.motor_cmd[i]
        c.q, c.dq, c.tau, c.kp, c.kd = 0.0, 0.0, 0.0, 0.0, 8.0
    cmd.crc = crc.Crc(cmd)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        pub.Write(cmd)
        time.sleep(0.002)
    print(f"damping sent for {a.seconds:.1f} s (kp 0, kd 8; motors keep it after this)", flush=True)
    return 0


ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["check", "release", "ramp", "damp"])
ap.add_argument("--iface", default=os.environ.get("DWBC_IFACE", "enp13s0"))
ap.add_argument("--domain", type=int, default=int(os.environ.get("DWBC_DDS_DOMAIN", "0")))
ap.add_argument("--seconds", type=float, default=None)
ap.add_argument("--hz", type=float, default=200.0)
ap.add_argument("--kp-scale", type=float, default=1.0)
ap.add_argument("--json")
a = ap.parse_args()
if a.seconds is None:
    a.seconds = {"check": 2.0, "ramp": 3.0, "damp": 1.5}.get(a.cmd, 0.0)
ChannelFactoryInitialize(a.domain, a.iface)
sys.stdout.flush()
rc = {"check": check, "release": release, "ramp": ramp, "damp": damp}[a.cmd](a)
os._exit(rc)
