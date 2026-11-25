"""Offline MuJoCo check of the leg policy's stepping (no DDS / ROS / pid files: safe while the real chain runs).

Runs the real G1GearWbcPolicy class (Balance + Walk ONNX) on the 43-DoF scene, arms held at the default pose, PD at the
physics rate with the yaml MOTOR_KP / MOTOR_KD. Two experiments:

  pulse   the operator's problem: navigate vx = V for T seconds (0.1 s ramps, like the interpolated chain), from
          different standing times -> number of foot lift-offs (ground truth: foot height in MuJoCo)
  step    SingleStep (dwbc_step.py): N touchdowns per request, from different standing times -> steps, distance

    source ~/piper_ws/dwbc/dwbc_env.sh; python tests/step_sim.py pulse|step [--json out.json]
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from dwbc_step import PROFILES, SingleStep, foot_heights, foot_pose  # noqa: E402

import decoupled_wbc  # noqa: E402
from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy  # noqa: E402
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model  # noqa: E402

PKG = os.path.dirname(decoupled_wbc.__file__)
WBC = yaml.safe_load(open(os.path.join(PKG, "control/main/teleop/configs/g1_29dof_gear_wbc.yaml")))
SCENE = os.path.join(PKG, "control/robot_model/model_data/g1/scene_43dof.xml")
GEAR_CFG = os.path.join(PKG, "..", WBC["GEAR_WBC_CONFIG"])
MODEL_PATH = "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"


class Sim:
    def __init__(self, seed=0):
        self.rm = instantiate_g1_robot_model(waist_location="lower_body")
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.sub = int(round(0.02 / self.dt))
        names = self.rm.joint_names
        # robot_model joint i <-> MuJoCo qpos / qvel address and actuator
        self.qadr, self.vadr, self.act = [], [], []
        for n in names:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.qadr.append(self.m.jnt_qposadr[j])
            self.vadr.append(self.m.jnt_dofadr[j])
            a = [k for k in range(self.m.nu) if self.m.actuator_trnid[k, 0] == j]
            self.act.append(a[0])
        self.qadr, self.vadr, self.act = map(np.array, (self.qadr, self.vadr, self.act))
        self.body = np.asarray(self.rm.get_joint_group_indices("body"))       # 29, motor order
        self.lower = np.asarray(self.rm.get_joint_group_indices("lower_body"))
        self.kp = np.full(len(names), 5.0)
        self.kd = np.full(len(names), 0.2)
        self.kp[self.body] = WBC["MOTOR_KP"]
        self.kd[self.body] = WBC["MOTOR_KD"]
        # real-robot conditions (env): STEP_SIM_WAIST_KD (real control loop: MOTOR_KD[14] - 10 = -5),
        # STEP_SIM_DELAY (control ticks of actuation delay)
        if os.environ.get("STEP_SIM_WAIST_KD"):
            self.kd[self.body[14]] = float(os.environ["STEP_SIM_WAIST_KD"])
        self.delay = int(os.environ.get("STEP_SIM_DELAY", "0"))
        self._tq = []
        self.q_tgt = np.asarray(self.rm.default_body_pose, dtype=float).copy()
        self.pol = G1GearWbcPolicy(robot_model=self.rm, config=GEAR_CFG, model_path=MODEL_PATH)
        self.pol.use_teleop_policy_cmd = True
        self.pol.use_policy_action = True
        self.q_tgt[self.lower] = self.pol.config["default_angles"]
        self.d.qpos[7 + 0:] = 0
        self.d.qpos[self.qadr] = self.q_tgt
        self.d.qpos[0:3] = [0, 0, 0.80]
        self.d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.t = 0.0
        self.nav = np.zeros(3)
        self.height = 0.74
        self.arms = np.asarray(self.rm.get_joint_group_indices("arms"))

    def obs(self):
        d = self.d
        return {
            "q": d.qpos[self.qadr].copy(), "dq": d.qvel[self.vadr].copy(),
            "floating_base_pose": d.qpos[0:7].copy(), "floating_base_vel": d.qvel[0:6].copy(),
        }

    def feet_world_z(self):
        return [float(self.d.xpos[f][2]) for f in self.fid]

    def feet_world_xy(self):
        return [self.d.xpos[f][:2].copy() for f in self.fid]

    def tick(self):
        """one 50 Hz control step"""
        o = self.obs()
        self.pol.set_observation(o)
        act = self.pol.get_action(self.t, o["q"][self.arms], np.array([self.height]), np.zeros(3), self.nav.copy())
        self._tq.append(np.asarray(act["body_action"][0][: len(self.lower)], dtype=float).copy())
        if len(self._tq) > self.delay:
            self.q_tgt[self.lower] = self._tq.pop(0)
        for _ in range(self.sub):
            q = self.d.qpos[self.qadr]
            dq = self.d.qvel[self.vadr]
            tau = self.kp * (self.q_tgt - q) - self.kd * dq
            self.d.ctrl[self.act] = tau
            mujoco.mj_step(self.m, self.d)
        self.t += 0.02
        return o

    def tilt_deg(self):
        w, x, y, z = self.d.qpos[3:7]
        g = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        return float(np.degrees(np.arccos(np.clip(g[2], -1, 1))))


class LiftCounter:
    """ground truth: a lift-off = foot > rest height + 2 cm, landing = back below rest + 0.8 cm"""

    def __init__(self, z0):
        self.z0 = list(z0)
        self.up = [False, False]
        self.lifts = [0, 0]
        self.seq = []

    def update(self, t, z):
        for i in (0, 1):
            if not self.up[i] and z[i] > self.z0[i] + 0.02:
                self.up[i] = True
                self.lifts[i] += 1
                self.seq.append(("LR"[i], round(t, 2)))
            elif self.up[i] and z[i] < self.z0[i] + 0.008:
                self.up[i] = False


def stand(sim, seconds):
    for _ in range(int(seconds / 0.02)):
        sim.tick()


def ramp_nav(t, t_on, T, v, ramp=0.1):
    if t < t_on or t > t_on + T + ramp:
        return 0.0
    a = min(1.0, (t - t_on) / ramp, (t_on + T + ramp - t) / ramp)
    return v * max(0.0, a)


def run_pulse(T, v, wait, seed):
    sim = Sim(seed)
    stand(sim, 2.0 + wait)
    lc = LiftCounter(sim.feet_world_z())
    base0 = sim.d.qpos[0:2].copy()
    t_on = sim.t + 0.05
    maxtilt = 0.0
    while sim.t < t_on + T + 3.0:
        sim.nav[0] = ramp_nav(sim.t, t_on, T, v)
        sim.tick()
        lc.update(sim.t, sim.feet_world_z())
        maxtilt = max(maxtilt, sim.tilt_deg())
    return {"T": T, "v": v, "wait": wait, "lifts": sum(lc.lifts), "LR": lc.lifts, "seq": lc.seq,
            "dx_cm": round(100 * float(sim.d.qpos[0] - base0[0]), 1), "maxtilt": round(maxtilt, 1),
            "fell": bool(sim.d.qpos[2] < 0.5)}


def run_step(direction, v, n_td, wait, seed, repeats=1, stop_on="touchdown"):
    sim = Sim(seed)
    stand(sim, 2.0 + wait)
    lc = LiftCounter(sim.feet_world_z())
    ss = (SingleStep(profiles=PROFILES, verbose=False) if stop_on == "profile" else
          SingleStep(speed=v, back_speed=v, touchdowns=n_td, stop_on=stop_on, verbose=False))
    base0 = sim.d.qpos[0:2].copy()
    w, x, y, z = sim.d.qpos[3:7]
    yaw0 = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    maxtilt = 0.0
    req_t = [sim.t + 0.05 + k * 2.5 for k in range(repeats)]
    t_end = req_t[-1] + 4.0
    o = sim.obs()
    while sim.t < t_end:
        if req_t and sim.t >= req_t[0]:
            ss.request(sim.t, direction)
            req_t.pop(0)
        zl, zr, fdx = foot_pose(sim.rm, o["q"], o["floating_base_pose"][3:7])
        nav = ss.update(sim.t, zl, zr, fdx)
        sim.nav[:] = nav if nav is not None else 0.0
        o = sim.tick()
        lc.update(sim.t, sim.feet_world_z())
        maxtilt = max(maxtilt, sim.tilt_deg())
    fx = sim.feet_world_xy()
    w, x, y, z = sim.d.qpos[3:7]
    yaw1 = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return {"dyaw_deg": round((yaw1 - yaw0 + 180) % 360 - 180, 1), "dir": direction, "v": v, "touchdowns": n_td, "stop_on": stop_on, "wait": wait, "repeats": repeats,
            "lifts": sum(lc.lifts), "LR": lc.lifts, "seq": lc.seq,
            "dx_cm": round(100 * float(sim.d.qpos[0] - base0[0]), 1),
            "dy_cm": round(100 * float(sim.d.qpos[1] - base0[1]), 1),
            "feet_dx_cm": round(100 * float(fx[0][0] - fx[1][0]), 1),
            "events": [(round(t, 2), e, s) for t, e, s in ss.log],
            "summaries": [{k: v for k, v in r.items()} for r in ss.summaries],
            "maxtilt": round(maxtilt, 1), "fell": bool(sim.d.qpos[2] < 0.5)}


def run_turn(script, wait=0.0, step_deg=15.0, speed=0.5, hold_speed=0.5, coast=None, t_total=None):
    """script: list of (t_offset, action) - ("flick", +1/-1) or ("hold", v) with v = 0 to release, sent every tick
    from that time on (like the teleop at 20-50 Hz). Returns the yaw track and the stop yaws."""
    from dwbc_step import yaw_from_quat_wxyz
    sim = Sim()
    stand(sim, 2.0 + wait)
    prof = dict(PROFILES, turn_left=(speed, "lift", 2), turn_right=(speed, "lift", 2))
    ss = SingleStep(profiles=prof, verbose=False, turn={"step_deg": step_deg, "hold_speed": hold_speed})
    if coast is not None:
        ss.COAST = coast
    lc = LiftCounter(sim.feet_world_z())
    t0 = sim.t
    yaw0 = yaw_from_quat_wxyz(sim.d.qpos[3:7])
    o = sim.obs()
    hold = None
    script = sorted(script)
    t_end = t0 + (t_total or (script[-1][0] + 4.0))
    track = []
    maxtilt = 0.0
    while sim.t < t_end:
        while script and sim.t - t0 >= script[0][0]:
            _, (kind, v) = script.pop(0)
            if kind == "flick":
                ss.request(sim.t, "turn_left" if v > 0 else "turn_right")
            else:
                hold = v
        if hold is not None:
            ss.set_hold(sim.t, hold)
            if hold == 0:
                hold = None
        zl, zr, fdx = foot_pose(sim.rm, o["q"], o["floating_base_pose"][3:7])
        nav = ss.update(sim.t, zl, zr, fdx, yaw=yaw_from_quat_wxyz(o["floating_base_pose"][3:7]))
        sim.nav[:] = nav if nav is not None else 0.0
        o = sim.tick()
        lc.update(sim.t, sim.feet_world_z())
        maxtilt = max(maxtilt, sim.tilt_deg())
        track.append((round(sim.t - t0, 2), round(float(np.degrees(yaw_from_quat_wxyz(sim.d.qpos[3:7]) - yaw0)), 1)))
    fx = sim.feet_world_xy()
    return {"yaw_deg": track[-1][1], "lifts": sum(lc.lifts), "seq": lc.seq, "summaries": ss.summaries,
            "events": [(round(t - t0, 2), e, s) for t, e, s in ss.log], "track": track,
            "feet_dx_cm": round(100 * float(fx[0][0] - fx[1][0]), 1), "maxtilt": round(maxtilt, 1),
            "fell": bool(sim.d.qpos[2] < 0.5)}


def run_squat(h_low, r_down, r_up, hold=3.0, wait=0.0, up_profile="linear"):
    """height 0.74 -> h_low at r_down m/s, hold, back up at r_up m/s ("linear" or "smooth" = smoothstep)."""
    sim = Sim()
    stand(sim, 2.0 + wait)
    z0 = sim.feet_world_z()
    b0 = sim.d.qpos[0:2].copy()
    while sim.height > h_low:
        sim.height = max(h_low, sim.height - r_down * 0.02)
        sim.tick()
    stand(sim, hold)
    lc = LiftCounter(z0)
    b1 = sim.d.qpos[0:2].copy()
    pitch_max = 0.0
    T = (0.74 - h_low) / r_up
    t_up = sim.t
    while sim.t < t_up + T + 3.0:
        a = min(1.0, (sim.t - t_up) / T)
        if up_profile == "smooth":
            a = a * a * (3 - 2 * a)
        sim.height = h_low + a * (0.74 - h_low)
        sim.tick()
        lc.update(sim.t, sim.feet_world_z())
        w, x, y, z = sim.d.qpos[3:7]
        pitch_max = max(pitch_max, abs(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))))
    return {"h_low": h_low, "r_down": r_down, "r_up": r_up, "profile": up_profile, "lifts": sum(lc.lifts),
            "seq": lc.seq, "dx_squat_cm": round(100 * float(b1[0] - b0[0]), 1),
            "dx_rise_cm": round(100 * float(sim.d.qpos[0] - b1[0]), 1), "pitch_max": round(pitch_max, 1),
            "fell": bool(sim.d.qpos[2] < 0.4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pulse", "step", "idle", "squat", "turn"])
    ap.add_argument("--lows", default="0.6,0.5,0.4")
    ap.add_argument("--rups", default="0.2,0.1,0.05,0.025")
    ap.add_argument("--profile", default="linear")
    ap.add_argument("--json")
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--waits", default="0,0.13,0.27,0.41,0.55")
    ap.add_argument("--Ts", default="0.2,0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--dirs", default="fwd,back,left,right")
    ap.add_argument("--td", default="1,2")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--stop-on", default="touchdown")
    ap.add_argument("--vs", default=None, help="comma list of speeds (overrides --v)")
    a = ap.parse_args()
    waits = [float(x) for x in a.waits.split(",")]
    out = []
    if a.mode == "turn":
        coast = float(os.environ["STEP_SIM_COAST"]) if os.environ.get("STEP_SIM_COAST") else None
        cases = [("flick L x1", [(0.05, ("flick", 1))]),
                 ("flick R x1", [(0.05, ("flick", -1))]),
                 ("flick L x2", [(0.05, ("flick", 1)), (0.35, ("flick", 1))]),
                 ("flick L x3", [(0.05, ("flick", 1)), (0.30, ("flick", 1)), (0.6, ("flick", 1))]),
                 ("flick R x4", [(0.05, ("flick", -1)), (0.3, ("flick", -1)), (0.6, ("flick", -1)), (0.9, ("flick", -1))]),
                 ("L then R", [(0.05, ("flick", 1)), (0.6, ("flick", -1))]),
                 ("hold L 1.0 2s", [(0.05, ("hold", 1.0)), (2.05, ("hold", 0))]),
                 ("hold R 0.5 3s", [(0.05, ("hold", -0.5)), (3.05, ("hold", 0))]),
                 ("hold L 0.3 1s", [(0.05, ("hold", 0.3)), (1.05, ("hold", 0))]),
                 ("flick+hold L", [(0.05, ("flick", 1)), (0.45, ("hold", 1.0)), (2.45, ("hold", 0))])]
        for name, sc in cases:
            rs = [run_turn(list(sc), wait=w, coast=coast) for w in waits[:3]]
            out += [dict(r, case=name, track=None) for r in rs]
            descr = []
            for r in rs:
                rel = [s for s in sc if s[1][0] == "hold" and s[1][1] == 0]
                yaw_rel = None
                if rel:
                    tr = rel[0][0]
                    yaw_rel = next(y for t, y in r["track"] if t >= tr)
                tgt = r["summaries"][-1].get("target_deg") if r["summaries"] else None
                descr.append(f"{r['yaw_deg']:+.1f}" + (f"(tgt {tgt:+.0f})" if tgt is not None and "flick" in name and "hold" not in name else "")
                             + (f"(at release {yaw_rel:+.1f}, +{abs(r['yaw_deg'] - yaw_rel):.1f} after)" if yaw_rel is not None else "")
                             + f" lifts {r['lifts']} extra {sum(s['extra'] for s in r['summaries'])}"
                             + (" FELL" if r["fell"] else ""))
            print(f"{name:14s}: " + " | ".join(descr), flush=True)
    elif a.mode == "squat":
        for h in [float(x) for x in a.lows.split(",")]:
            for ru in [float(x) for x in a.rups.split(",")]:
                rs = [run_squat(h, 0.2, ru, wait=w, up_profile=a.profile) for w in waits[:3]]
                out += rs
                print(f"squat to {h:.2f} m, rise {ru:.3f} m/s ({a.profile}): lifts {[r['lifts'] for r in rs]}"
                      f"  first {[r['seq'][:1] for r in rs]}  dx while squatting {[r['dx_squat_cm'] for r in rs]}"
                      f"  dx while rising {[r['dx_rise_cm'] for r in rs]}  pitch_max {max(r['pitch_max'] for r in rs)}"
                      f"  fell {any(r['fell'] for r in rs)}", flush=True)
    elif a.mode == "idle":
        r = run_pulse(0.0, 0.0, 0.0, 0)
        print(r)
        out.append(r)
    elif a.mode == "pulse":
        for T in [float(x) for x in a.Ts.split(",")]:
            rs = [run_pulse(T, a.v, w, 0) for w in waits]
            out += rs
            print(f"T={T:.1f}s v={a.v}: lifts {[r['lifts'] for r in rs]}  dx_cm {[r['dx_cm'] for r in rs]}"
                  f"  maxtilt {max(r['maxtilt'] for r in rs)}  fell {any(r['fell'] for r in rs)}", flush=True)
    else:
        for v in ([float(x) for x in a.vs.split(",")] if a.vs else [a.v]):
          for n in [int(x) for x in a.td.split(",")]:
            for dr in a.dirs.split(","):
                rs = [run_step(dr, v, n, w, 0, a.repeats, a.stop_on) for w in waits]
                out += rs
                print(f"{a.stop_on}={n} {dr:5s} v={v}: lifts {[r['lifts'] for r in rs]}  dx {[r['dx_cm'] for r in rs]}"
                      f"  dy {[r['dy_cm'] for r in rs]}  feet_dx {[r['feet_dx_cm'] for r in rs]}"
                      f"  maxtilt {max(r['maxtilt'] for r in rs)}  fell {any(r['fell'] for r in rs)}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
