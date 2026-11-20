"""Gripper path check without the robot (run inside `unshare -rn`, lo up): Dex1 state -> dex3_emu -> WBC hand.

Needs dex3_emu.py --iface lo --sides left,right running and the WBC hand classes. Publishes a fake Dex1 position
(left: open -> closed, right: open), then
  1. runs the upstream G1ThreeFingerHand.calibrate_hand() against the emulated state (must end, offset ~0)
  2. reads the emulated Dex3 state: closure must follow the Dex1 position
  3. computes the closure the Dex1 bridge would take from the WBC gripper-IK closed command (dex3_closure)
"""
import json
import os
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

ChannelFactoryInitialize(0, "lo")
from decoupled_wbc.control.envs.g1.g1_hand import G1ThreeFingerHand  # noqa: E402
from decoupled_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver  # noqa: E402

calib = json.load(open(os.path.expanduser("~/g1_hub/config.json")))["gripper_calib"]
q_dex1 = {"left": calib["left"]["q_open"], "right": calib["right"]["q_open"]}
pubs = {s: ChannelPublisher(f"rt/dex1/{s}/state", MotorStates_) for s in ("left", "right")}
for p in pubs.values():
    p.Init()
stop = threading.Event()


def feed():
    while not stop.is_set():
        for s, p in pubs.items():
            st = unitree_go_msg_dds__MotorState_()
            st.q = float(q_dex1[s])
            p.Write(MotorStates_(states=[st]))
        time.sleep(0.01)


threading.Thread(target=feed, daemon=True).start()
time.sleep(1.0)
res = {}
hands = {s: G1ThreeFingerHand(is_left=(s == "left")) for s in ("left", "right")}
time.sleep(0.5)
for s, h in hands.items():
    t0 = time.monotonic()
    h.calibrate_hand()
    res[f"{s}_calibration"] = {"seconds": round(time.monotonic() - t0, 2),
                               "q0_offset_deg": round(float(np.degrees(h.hand_q_offset[0])), 2)}


def closure_of(q, side):  # same mapping as dex3_emu (inverse) / sonic_dex1_bridge
    ref = np.array([1.0, 1.5, 1.0, 1.5])
    sign = -1.0 if side == "left" else 1.0
    return float(np.clip(sign * np.asarray(q[3:7]) / ref, 0.0, 1.0).mean())


for frac in (0.0, 0.5, 1.0):
    q_dex1["left"] = calib["left"]["q_open"] + frac * (calib["left"]["q_closed"] - calib["left"]["q_open"])
    time.sleep(0.3)
    hq = hands["left"].observe()["hand_q"]
    res[f"left_dex1_{frac:.1f}"] = {"emulated_hand_q": np.round(hq, 2).tolist(),
                                     "closure_seen_by_bridge": round(closure_of(hq, "left"), 2)}
for s in ("left", "right"):
    ik = G1GripperInverseKinematicsSolver(s)
    f = np.zeros((25, 4, 4)); f[4, 0, 3] = 1.0
    open_q = ik({"position": f})
    f[9, 0, 3] = 1.0
    closed_q = ik({"position": f})
    res[f"{s}_ik_command_closure"] = {"open": round(closure_of(open_q, s), 2), "closed": round(closure_of(closed_q, s), 2)}
stop.set()
print(json.dumps(res, indent=1), flush=True)
ok = (all(abs(res[f"{s}_calibration"]["q0_offset_deg"]) < 1 for s in ("left", "right"))
      and res["left_dex1_0.0"]["closure_seen_by_bridge"] < 0.05 and res["left_dex1_1.0"]["closure_seen_by_bridge"] > 0.8
      and all(res[f"{s}_ik_command_closure"]["closed"] > 0.8 and res[f"{s}_ik_command_closure"]["open"] < 0.05
              for s in ("left", "right")))
print("PASS" if ok else "FAIL", flush=True)
os._exit(0 if ok else 1)
