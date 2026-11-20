"""Dex3 hand-state emulator for a G1 with Dex1-1 grippers (decoupled-WBC real chain).

The WBC control loop with hands on reads rt/dex3/<side>/state (HandState_, 7 joints) every cycle (no state -> crash)
and calibrates the thumb at start (rotates joint 0 until it stops). Our robot has Dex1 grippers, so this node
publishes a Dex3 state per side at 100 Hz, made from the Dex1 position on rt/dex1/<side>/state:
  joint 0      fixed at +60 deg: the calibration then sees a stopped thumb and computes a zero offset
  joints 1..6  closure * the gripper-IK "index close" pose (closure 0 = open .. 1 = closed, from gripper_calib)
so the recorded hand state matches the hand commands the teleop IK sends. The commands themselves
(rt/dex3/<side>/cmd) are turned into Dex1 motion by ~/g1_bridge/sonic_dex1_bridge.py, as with SONIC.
A side without Dex1 state is published as an open hand (and reported).
"""
import argparse
import json
import os
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

# G1GripperInverseKinematicsSolver._get_index_close_q_desired (left; right = negated)
CLOSE_L = np.array([-0.5, 0.7, 0.7, -1.5, -1.5, -0.6, -1.5])
THUMB_CAL = np.deg2rad(60.0)

ap = argparse.ArgumentParser()
ap.add_argument("--sides", default="left,right")
ap.add_argument("--iface", default="enp13s0")
ap.add_argument("--domain", type=int, default=0)
ap.add_argument("--calib", default=os.path.expanduser("~/g1_hub/config.json"))
ap.add_argument("--hz", type=float, default=100.0)
a = ap.parse_args()

calib = (json.load(open(a.calib)).get("gripper_calib") or {}) if os.path.exists(a.calib) else {}
ChannelFactoryInitialize(a.domain, a.iface)
last = {}
subs, pubs, msgs = {}, {}, {}


def on_dex1(m, side):
    if m.states:
        last[side] = (time.monotonic(), float(m.states[0].q))


for side in ("left", "right"):
    if side in a.sides.split(","):
        subs[side] = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
        subs[side].Init(lambda m, s=side: on_dex1(m, s), 10)
    pubs[side] = ChannelPublisher(f"rt/dex3/{side}/state", HandState_)
    pubs[side].Init()
    msgs[side] = unitree_hg_msg_dds__HandState_()


def closure(side):
    c = calib.get(side) or {}
    if side not in last or "q_open" not in c or time.monotonic() - last[side][0] > 0.5:
        return 0.0
    o, cl = float(c["q_open"]), float(c["q_closed"])
    return float(np.clip((last[side][1] - o) / (cl - o), 0.0, 1.0))


print(f"dex3_emu: publishing rt/dex3/{{left,right}}/state at {a.hz:.0f} Hz from Dex1 {a.sides}; calib {calib}",
      flush=True)
t_next, t_report = time.monotonic(), 0.0
while True:
    for side in ("left", "right"):
        c = closure(side)
        q = c * (CLOSE_L if side == "left" else -CLOSE_L)
        q[0] = THUMB_CAL
        m = msgs[side]
        for i in range(7):
            m.motor_state[i].q = float(q[i])
            m.motor_state[i].dq = 0.0
            m.motor_state[i].tau_est = 0.0
        pubs[side].Write(m)
    now = time.monotonic()
    if now - t_report > 5.0:
        print("  " + "  ".join(f"{s}: dex1 q={last[s][1]:+.2f} closure={closure(s):.2f}" if s in last else f"{s}: no dex1"
                               for s in ("left", "right")), flush=True)
        t_report = now
    t_next += 1.0 / a.hz
    time.sleep(max(0.0, t_next - time.monotonic()))
