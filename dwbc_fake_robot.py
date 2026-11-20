"""Stand-in G1 for dry runs of the REAL decoupled-WBC chain: the upstream MuJoCo sim as a separate process.

Run it inside a private network namespace whose lo also carries 192.168.123.99, so that `--interface real` in the
control loop resolves to lo with env_type "real" (the real code path: hand calibration, real kd, sys.exit on a
safety violation) while nothing can reach the robot or the other sessions' DDS on the host:

    unshare -rn bash -c 'ip link set lo up; ip addr add 192.168.123.99/24 dev lo; ...'

Publishes rt/lowstate, rt/secondary_imu, rt/dex3/*/state (sim Dex3 hands) on lo / domain 0 and takes rt/lowcmd.
The elastic band (the gantry) starts ON; `touch /tmp/dwbc_fake_band` toggles it. Logs the true base height and tilt
once per second (the real-mode control loop has no odometry).
"""
import os
import time

import numpy as np

import decoupled_wbc.control.main.teleop.configs.configs as C
from decoupled_wbc.control.envs.g1.sim.base_sim import BaseSimulator

cfg = C.ControlLoopConfig(interface="sim", enable_onscreen=False)
config = cfg.load_wbc_yaml()
config.update({"INTERFACE": "lo", "DOMAIN_ID": 0, "ENV_TYPE": "sim", "ENABLE_ONSCREEN": False,
               "ENABLE_OFFSCREEN": False})
sim = BaseSimulator(config)
sim.start_as_thread()

# what the robot is being told: last rt/lowcmd gains (shows damping after an e-stop: kp 0, kd 8)
from unitree_sdk2py.core.channel import ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_  # noqa: E402

last_cmd = {"m": None, "n": 0}


def _on_cmd(m):
    last_cmd["m"] = m
    last_cmd["n"] += 1


_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
_cmd_sub.Init(_on_cmd, 10)
BAND = "/tmp/dwbc_fake_band"
LOWER = "/tmp/dwbc_fake_lower"
SLACK = "/tmp/dwbc_fake_slack"


class Gantry:
    """The upstream band is a stiff two-sided spring holding the pelvis at 1 + length m (feet in the air).
    lower(): over 3 s bring the hold point down until the feet carry weight (band still taut, like a gantry lowered
    until the feet touch). slack(): switch to a rope that only pulls when the pelvis sinks below z_slack."""

    def __init__(self, band):
        self.band, self.orig = band, band.Advance
        self.mode, self.t0, self.z_slack = "stiff", 0.0, 0.62
        band.Advance = self.advance

    def lower(self):
        self.mode, self.t0 = "lowering", time.monotonic()

    def slack(self):
        self.mode = "slack"
        print(time.strftime("%H:%M:%S"), f"gantry: rope slack (catches below {self.z_slack} m)", flush=True)

    def advance(self, pose):
        if self.mode == "lowering":
            s = min(1.0, (time.monotonic() - self.t0) / 3.0)
            self.band.length = -0.21 * s
            if s >= 1.0:
                self.mode = "stiff"
                print(time.strftime("%H:%M:%S"), "gantry: feet on the ground, band still taut", flush=True)
        if self.mode == "slack":
            pos, vel = pose[0:3], pose[7:10]
            f = np.zeros(6)
            if pos[2] < self.z_slack:
                f[2] = 3000.0 * (self.z_slack - pos[2]) - 300.0 * vel[2]
                f[0:2] = -200.0 * pos[0:2] - 50.0 * vel[0:2]
            return f
        return self.orig(pose)


gantry = Gantry(sim.sim_env.elastic_band)
print("fake robot up (lo, domain 0); band ON (feet in the air); touch", LOWER, "to lower onto the feet,", BAND,
      "to toggle the band", flush=True)
t_log = 0.0
while True:
    if os.path.exists(BAND):
        os.remove(BAND)
        sim.sim_env.handle_keyboard_button("9")
        print(time.strftime("%H:%M:%S"), "band toggled", flush=True)
    if os.path.exists(LOWER):
        os.remove(LOWER)
        gantry.lower()
        print(time.strftime("%H:%M:%S"), "gantry lowering (3 s)", flush=True)
    if os.path.exists(SLACK):
        os.remove(SLACK)
        gantry.slack()
    now = time.time()
    if now - t_log >= 1.0:
        d = sim.sim_env.mj_data
        w, x, y, z = d.qpos[3:7]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        m = last_cmd["m"]
        gains = "no lowcmd" if m is None else (
            f"lowcmd {last_cmd['n']}/s kp_sum {sum(c.kp for c in m.motor_cmd[:29]):.0f} "
            f"kd_mean {np.mean([c.kd for c in m.motor_cmd[:29]]):.1f}")
        last_cmd["n"] = 0
        print(f"{time.strftime('%H:%M:%S')} base z {d.qpos[2]:.3f} m  tilt {tilt:4.1f} deg  "
              f"xy {d.qpos[0]:+.2f} {d.qpos[1]:+.2f}  {gains}", flush=True)
        t_log = now
    if sim.sim_thread is not None and not sim.sim_thread.is_alive():
        print("sim thread died", flush=True)
        os._exit(1)
    time.sleep(0.05)
