"""Full-chain sim test of single-step mode: needs `DWBC_STEP=1 dwbc.sh sim` (MuJoCo over DDS, control loop over ROS).
Plays the teleop loop: publishes goals (upper body held, navigate 0) with control_data dwbc_step {seq, dir} exactly as
the PICO streamer patch does, plus keyboard 't'. Prints base displacement per request; the lift / touchdown events are
in the control loop log ([dwbc_step] lines).   python tests/step_chain_test.py"""
import os
import threading
import time

import numpy as np
from std_msgs.msg import String

from decoupled_wbc.control.main.constants import (CONTROL_GOAL_TOPIC, DEFAULT_BASE_HEIGHT, KEYBOARD_INPUT_TOPIC,
                                                  STATE_TOPIC_NAME)
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber

ros = ROSManager(node_name="dwbc_step_chain_test")
goal_pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
key_pub = ros.node.create_publisher(String, KEYBOARD_INPUT_TOPIC, 10)
state_sub = ROSMsgSubscriber(STATE_TOPIC_NAME)
rm = instantiate_g1_robot_model(  # must match the control loop (waist in the upper body with --enable-waist)
    waist_location="lower_and_upper_body" if os.environ.get("DWBC_ENABLE_WAIST") == "1" else "lower_body")
UP = np.asarray(rm.get_joint_group_indices("upper_body"))
last = {"m": None}


def sampler():
    while True:
        m = state_sub.get_msg()
        if m is not None:
            last["m"] = m
        time.sleep(0.005)


threading.Thread(target=sampler, daemon=True).start()
t0 = time.monotonic()
while last["m"] is None:
    if time.monotonic() - t0 > 20:
        raise SystemExit("no state: is `DWBC_STEP=1 dwbc.sh sim` running?")
    time.sleep(0.1)
q_up = np.asarray(last["m"]["q"], dtype=float)[UP].copy()
step = {"seq": 0, "dir": None, "mode": True}


def key(k):
    m = String()
    m.data = k
    key_pub.publish(m)
    print(f"  key {k!r}", flush=True)


def base():
    return np.asarray(last["m"]["floating_base_pose"], dtype=float)


def publish(dur, first=False):
    ts = time.monotonic()
    while time.monotonic() - ts < dur:
        now = time.monotonic()
        goal_pub.publish({"target_upper_body_pose": q_up, "navigate_cmd": [0.0, 0.0, 0.0],
                          "base_height_command": DEFAULT_BASE_HEIGHT, "timestamp": now,
                          "target_time": now + (2.0 if first else 0.05), "dwbc_step": dict(step)})
        if first:
            time.sleep(2.0)
            first = False
        time.sleep(0.05)


publish(3, first=True)
key("]")
publish(2)
key("9")
publish(4)
res = []
for d in ["fwd", "back", "left", "right", "fwd", "fwd"]:
    b0 = base()
    step["seq"] += 1
    step["dir"] = d
    print(f"PICO-path step #{step['seq']} {d}", flush=True)
    publish(3.0)
    b1 = base()
    res.append((d, "pico", 100 * (b1[:2] - b0[:2]), b1[2]))
b0 = base()
key("t")
publish(3.0)
b1 = base()
res.append(("fwd", "key t", 100 * (b1[:2] - b0[:2]), b1[2]))
for d, src, dxy, z in res:
    print(f"{src:6s} {d:5s}: moved x {dxy[0]:+5.1f} y {dxy[1]:+5.1f} cm, pelvis z {z:.2f}", flush=True)


# ---- turning (2026-11-24): flicks by angle (added up), held stick = continuous, release = stop
def yaw_deg():
    w, x, y, z = base()[3:7]
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def wrap(a):
    return (a + 180) % 360 - 180


turns = []
for name, flicks in (("flick L x1", ["turn_left"]), ("flick R x2", ["turn_right", "turn_right"]),
                     ("flick L x3", ["turn_left"] * 3)):
    y0 = yaw_deg()
    for d in flicks:
        step["seq"] += 1
        step["dir"] = d
        publish(0.3)
    publish(4.0)
    turns.append((name, 15 * len(flicks) * (1 if "L" in name else -1), wrap(yaw_deg() - y0)))
y0 = yaw_deg()
step["turn_hold"] = 1.0
publish(2.0)
y_rel = yaw_deg()
step["turn_hold"] = 0.0
publish(3.0)
turns.append(("hold L 2 s", None, wrap(yaw_deg() - y0), wrap(yaw_deg() - y_rel)))
step.pop("turn_hold")
for t in turns:
    if t[1] is not None:
        print(f"turn {t[0]:11s}: target {t[1]:+4d} deg, turned {t[2]:+5.1f} deg", flush=True)
    else:
        print(f"turn {t[0]:11s}: turned {t[2]:+5.1f} deg, {t[3]:+4.1f} of it after the release", flush=True)
print(f"pelvis z at the end {base()[2]:.2f}", flush=True)
