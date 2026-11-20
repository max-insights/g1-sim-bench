"""Guard for the decoupled-WBC real chain: the only place that turns a stop into DAMPING.

Upstream the control loop just exits on a joint-velocity violation (sys.exit, no damping), and its 'o' key only
switches the leg policy off (the legs then hold their current angles). Exiting leaves the motors on their last
PD targets. This guard watches, at 50 Hz:
  - the control loop's process group (pid file): it died           -> damping
  - the flag file (default /tmp/dwbc_estop): `dwbc_real.sh estop`   -> kill the control loop, damping
  - motor temperatures (rt/lowstate): casing >= 85 C or winding >= 115 C for 2 s (Unitree SDK limits, as the SONIC
    watchdog) -> kill the control loop, damping; warning by voice at 75 / 100 C
Damping = kp 0 / kd 8 on all 29 motors for 1.5 s (the SONIC e-stop command); the motors keep it afterwards.
Publishes nothing until it stops. Writes --csv (1 Hz temperatures, ankle tau/q) and exits after the stop.
"""
import argparse
import csv
import os
import signal
import subprocess
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NAMES = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll", "R_hip_pitch",
         "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll", "waist_yaw", "waist_roll",
         "waist_pitch", "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist_roll",
         "L_wrist_pitch", "L_wrist_yaw", "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
         "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"]
N = 29
ANKLES = [4, 5, 10, 11]

ap = argparse.ArgumentParser()
ap.add_argument("--pidfile", required=True, help="control loop process-group id file")
ap.add_argument("--flag", default="/tmp/dwbc_estop")
ap.add_argument("--iface", default="enp13s0")
ap.add_argument("--domain", type=int, default=0)
ap.add_argument("--warn-casing", type=int, default=75)
ap.add_argument("--warn-winding", type=int, default=100)
ap.add_argument("--stop-casing", type=int, default=85)
ap.add_argument("--stop-winding", type=int, default=115)
ap.add_argument("--hold", type=float, default=2.0)
ap.add_argument("--csv")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def say(text):
    if a.quiet:
        return
    try:
        subprocess.Popen(["spd-say", "-l", "zh", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


ChannelFactoryInitialize(a.domain, a.iface)
state = {"m": None}
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(lambda m: state.__setitem__("m", m), 10)
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
crc = CRC()
pgid = int(open(a.pidfile).read().strip())
log(f"guard up: control loop pgid {pgid}, flag {a.flag}, stop casing {a.stop_casing} / winding {a.stop_winding} C")


def alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


def stop(reason):
    log(f"STOP: {reason}")
    if alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)  # no cleanup wanted: the next command must be damping
        except OSError:
            pass
    m = state["m"]
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = int(m.mode_machine) if m is not None else 5
    for i in range(N):
        c = cmd.motor_cmd[i]
        c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, 0.0, 0.0, 0.0, 0.0, 8.0
    cmd.crc = crc.Crc(cmd)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5:
        pub.Write(cmd)
        time.sleep(0.002)
    log("damping sent (kp 0, kd 8 on all motors)")
    say("全身控制已停止，电机阻尼")
    try:
        os.remove(a.flag)
    except OSError:
        pass


w = None
if a.csv:
    f = open(a.csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["unix_t", "hottest", "casing_max", "winding_max"] +
               [f"{NAMES[j]}_{k}" for j in ANKLES for k in ("casing", "winding", "tau", "q")])
hot_since, last_warn, last_row = None, 0.0, 0.0
try:
    os.remove(a.flag)
except OSError:
    pass
while True:
    if os.path.exists(a.flag):
        stop("e-stop requested (dwbc_real.sh estop)")
        break
    if not alive(pgid):
        stop("control loop exited")
        break
    m = state["m"]
    now = time.time()
    if m is not None:
        cas = np.array([m.motor_state[i].temperature[0] for i in range(N)])
        win = np.array([m.motor_state[i].temperature[1] for i in range(N)])
        j = int(np.argmax(np.maximum(cas - a.stop_casing, win - a.stop_winding)))
        if cas.max() >= a.stop_casing or win.max() >= a.stop_winding:
            hot_since = hot_since or now
            if now - hot_since >= a.hold:
                stop(f"motor temperature: {NAMES[j]} casing {cas[j]} C winding {win[j]} C")
                break
        else:
            hot_since = None
        if (cas.max() >= a.warn_casing or win.max() >= a.warn_winding) and now - last_warn > 20:
            jj = int(np.argmax(win))
            log(f"warning: hot motor {NAMES[jj]} casing {cas[jj]} winding {win[jj]}")
            say("电机温度偏高")
            last_warn = now
        if w is not None and now - last_row >= 1.0:
            jj = int(np.argmax(win))
            row = [round(now, 2), NAMES[jj], int(cas.max()), int(win.max())]
            for k in ANKLES:
                ms = m.motor_state[k]
                row += [int(ms.temperature[0]), int(ms.temperature[1]), round(float(ms.tau_est), 2), round(float(ms.q), 3)]
            w.writerow(row)
            f.flush()
            last_row = now
    time.sleep(0.02)
log("guard exit")
os._exit(0)
