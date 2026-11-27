"""Sim study: upper-body sway during single steps (DWBC leg policy, MuJoCo). Candidates are overrides, no defaults
changed. Usage: source dwbc_env.sh; STEP_SIM_DELAY=1 nice -n 19 python walk_sim.py CAND [CAND...] [--json out]
CAND "custom" takes its overrides from --params JSON (keys as in CANDS: speed_scale, ramp, hyst, lpf, waist_kd [yaw, roll,
pitch], waist_kp, height, rpy). Used by tests/sim_eval.py walk. 1 tick of delay matched the real pelvis roll (2026-11-26).
--upper NPZ (2026-11-27): during the steps the upper body follows a recorded IK output (names, t, q = an arm-IK replay
window): arm targets go to the arm PD, the waist becomes the torso rpy command of the leg policy exactly as upstream
(g1_decoupled_whole_body_policy: torso_link relative to the yaw-only pelvis), arms given to the policy = targets.
The arms first blend in from the default pose over 1.5 s while standing. Used by tests/sim_eval.py walkik."""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("STEP_SIM_WAIST_KD", "0")      # real: dwbc_gains.json waist_pitch_kd 0
import step_sim as S
from dwbc_step import SingleStep, foot_pose, yaw_from_quat_wxyz, load_profiles

PROF = load_profiles()[0]

CANDS = {
    "base": {},
    "slow": {"speed_scale": 0.7},                        # all step speeds x0.7
    "ramp": {"ramp": 0.6},                               # cmd rate limit (m/s per s), floor 0.06 to engage Walk
    "ramp_fast": {"ramp": 1.2},
    "hyst": {"hyst": 0.4},                               # keep the Walk net 0.4 s after the command drops to 0
    "hyst_ramp": {"hyst": 0.4, "ramp": 0.6},
    "lpf": {"lpf": 0.5},                                 # EMA on leg targets (alpha new)
    "waist_kd": {"waist_kd": [15, 15, 15]},              # waist yaw/roll/pitch kd (yaml 5/5/5, real pitch 0)
    "waist_kp": {"waist_kp": 400, "waist_kd": [8, 8, 8]},
    "low": {"height": 0.70},
    "pitch_fwd": {"rpy": [0, 0.05, 0]},
    "low72": {"height": 0.72},
    "wkd10": {"waist_kd": [10, 10, 10]},
    "wkd_yr15": {"waist_kd": [15, 15, 0]},
    "low72_wkd10": {"height": 0.72, "waist_kd": [10, 10, 10]},
    "lpf7": {"lpf": 0.7},
    "wkd_p10": {"waist_kd": [5, 5, 10]},
}


def euler(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return r, p


UP_RAMP = 1.5


def upper_player(sim, path):
    """q_full(t_play, alpha) for the arm-IK output in NPZ, and the upstream torso rpy for it."""
    from pinocchio import rpy as pin_rpy
    z = np.load(path)
    names = sim.rm.joint_names
    un = [str(v) for v in z["names"]]
    cols = [(names.index(n), i) for i, n in enumerate(un) if n in names]
    t, q = z["t"] - z["t"][0], z["q"]
    q0 = np.zeros(len(names))
    q0[sim.arms] = sim.q_tgt[sim.arms]

    def at(tp, alpha):
        qf = q0.copy()
        for j, i in cols:
            qf[j] = q0[j] + alpha * (np.interp(tp, t, q[:, i]) - q0[j])
        return qf

    def rpy(qf):
        sim.rm.cache_forward_kinematics(qf, auto_clip=False)
        tor = sim.rm.frame_placement("torso_link").rotation
        pel = sim.rm.frame_placement("pelvis").rotation
        yaw = np.arctan2(pel[1, 0], pel[0, 0])
        return np.asarray(pin_rpy.matrixToRpy(pin_rpy.rpyToMatrix(0, 0, yaw).T @ tor), float)
    return at, rpy


def run(cand, direction, wait, repeats=3, gap=2.2, upper=None):
    c = CANDS[cand]
    sim = S.Sim()
    rm = sim.rm
    names = rm.joint_names
    wi = [names.index(n) for n in ("waist_roll_joint", "waist_yaw_joint", "waist_pitch_joint")]
    if "waist_kd" in c:
        for j, k in zip([names.index("waist_yaw_joint"), names.index("waist_roll_joint"), names.index("waist_pitch_joint")], c["waist_kd"]):
            sim.kd[j] = k
    if "waist_kp" in c:
        for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            sim.kp[names.index(n)] = c["waist_kp"]
    if "height" in c:
        sim.height = c["height"]
    S.stand(sim, 2.5 + wait)
    up = upper_player(sim, upper) if upper else None
    t_up = sim.t + UP_RAMP
    prof = dict(PROF)
    if "speed_scale" in c:
        prof = {k: ((v[0] * c["speed_scale"],) + tuple(v[1:]) if isinstance(v, tuple) else v) for k, v in prof.items()}
    ss = SingleStep(profiles={k: v for k, v in prof.items() if isinstance(v, tuple)}, verbose=False,
                    turn={"step_deg": 15.0, "hold_speed": 0.5})
    lc = S.LiftCounter(sim.feet_world_z())
    p1 = sim.pol.policy_1
    rpy = np.array(c.get("rpy", [0, 0, 0]), float)
    cur = np.zeros(3)
    last_nz = -1e9
    filt = None
    rows = []
    req = [sim.t + 0.05 + (UP_RAMP if up else 0.0) + k * gap for k in range(repeats)]
    t_end = req[-1] + 3.5
    o = sim.obs()
    base0 = sim.d.qpos[0:3].copy()
    yaw0 = yaw_from_quat_wxyz(sim.d.qpos[3:7])
    fell = False
    while sim.t < t_end:
        if req and sim.t >= req[0]:
            ss.request(sim.t, direction)
            req.pop(0)
        zl, zr, fdx = foot_pose(rm, o["q"], o["floating_base_pose"][3:7])
        nav = ss.update(sim.t, zl, zr, fdx, yaw=yaw_from_quat_wxyz(o["floating_base_pose"][3:7]))
        tgt = np.array(nav if nav is not None else [0.0, 0.0, 0.0], float)
        if "ramp" in c:
            step = c["ramp"] * 0.02 * np.array([1, 1, 2.5])
            for i in range(3):
                if tgt[i] != 0 and abs(cur[i]) < 0.06:
                    cur[i] = 0.06 * np.sign(tgt[i])
                cur[i] += np.clip(tgt[i] - cur[i], -step[i], step[i])
                if tgt[i] == 0 and abs(cur[i]) < 0.06:
                    cur[i] = 0.0
        else:
            cur = tgt
        sim.nav[:] = cur
        if np.linalg.norm(cur) >= 0.05:
            last_nz = sim.t
        sim.pol.policy_1 = sim.pol.policy_2 if ("hyst" in c and sim.t - last_nz < c["hyst"]) else p1
        # tick with optional torso rpy cmd and leg-target filter
        ob = sim.obs()
        sim.pol.set_observation(ob)
        if up:
            qf = up[0](max(0.0, sim.t - t_up), min(1.0, 1.0 - (t_up - sim.t) / UP_RAMP))
            sim.q_tgt[sim.arms] = qf[sim.arms]
            act = sim.pol.get_action(sim.t, qf[sim.arms], np.array([sim.height]), rpy + up[1](qf), sim.nav.copy())
        else:
            act = sim.pol.get_action(sim.t, ob["q"][sim.arms], np.array([sim.height]), rpy, sim.nav.copy())
        a = np.asarray(act["body_action"][0][: len(sim.lower)], float).copy()
        if "lpf" in c:
            filt = a if filt is None else c["lpf"] * a + (1 - c["lpf"]) * filt
            a = filt
        sim._tq.append(a)
        if len(sim._tq) > sim.delay:
            sim.q_tgt[sim.lower] = sim._tq.pop(0)
        for _ in range(sim.sub):
            q = sim.d.qpos[sim.qadr]; dq = sim.d.qvel[sim.vadr]
            sim.d.ctrl[sim.act] = sim.kp * (sim.q_tgt - q) - sim.kd * dq
            S.mujoco.mj_step(sim.m, sim.d)
        sim.t += 0.02
        o = sim.obs()
        lc.update(sim.t, sim.feet_world_z())
        r, p = euler(sim.d.qpos[3:7])
        rows.append([sim.t, r, p, sim.d.qvel[3], sim.d.qvel[4]] + [sim.d.qpos[sim.qadr[i]] for i in wi])
        if sim.d.qpos[2] < 0.5:
            fell = True
            break
    R = np.array(rows)
    # per step: start -> idle (+0.8 s), like the real analysis
    ev, cur_s = [], None
    for t, e, s in ss.log:
        if e == "start":
            cur_s = t
        elif e == "idle" and cur_s is not None:
            ev.append((cur_s, t)); cur_s = None
    steps = []
    for (a0, b0), sm in zip(ev, ss.summaries):
        w = (R[:, 0] >= a0) & (R[:, 0] <= b0 + 0.8)
        X = R[w]
        steps.append(dict(roll=np.degrees(np.ptp(X[:, 1])), pitch=np.degrees(np.ptp(X[:, 2])),
                          roll_rate=np.degrees(np.abs(X[:, 3]).max()), pitch_rate=np.degrees(np.abs(X[:, 4]).max()),
                          w_roll=np.degrees(np.ptp(X[:, 5])), w_yaw=np.degrees(np.ptp(X[:, 6])), w_pitch=np.degrees(np.ptp(X[:, 7])),
                          dur=b0 - a0, extra=sm["extra"], lifts=len([x for x in sm["seq"] if x.isupper()])))
    w1, x1, y1, z1 = sim.d.qpos[3:7]
    return dict(cand=cand, dir=direction, wait=wait, fell=fell, steps=steps,
                dx=100 * float(sim.d.qpos[0] - base0[0]) / repeats, dy=100 * float(sim.d.qpos[1] - base0[1]) / repeats,
                dyaw=float(np.degrees((yaw_from_quat_wxyz(sim.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi)) / repeats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--dirs", default="back,left,right,fwd,turn_left")
    ap.add_argument("--waits", default="0,0.17,0.33")
    ap.add_argument("--json")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--upper", default=None, help="arm-IK output window (npz: names, t, q) to play during the steps")
    a = ap.parse_args()
    CANDS["custom"] = json.loads(a.params)
    out = []
    for cand in a.cands:
        for d in a.dirs.split(","):
            for w in [float(x) for x in a.waits.split(",")]:
                r = run(cand, d, w, upper=a.upper)
                out.append(r)
            st = [s for r in out if r["cand"] == cand and r["dir"] == d for s in r["steps"]]
            rr = [r for r in out if r["cand"] == cand and r["dir"] == d]
            m = lambda k: np.median([s[k] for s in st]) if st else float("nan")
            print(f"{cand:10s} {d:10s} n={len(st):2d} roll {m('roll'):5.1f} pitch {m('pitch'):5.1f} rr {m('roll_rate'):5.1f} "
                  f"pr {m('pitch_rate'):5.1f} | waist r/y/p {m('w_roll'):4.1f}/{m('w_yaw'):4.1f}/{m('w_pitch'):4.1f} | dur {m('dur'):4.2f} "
                  f"extra {sum(s['extra'] > 0 for s in st)}/{len(st)} | per step dx {np.mean([r['dx'] for r in rr]):5.1f} "
                  f"dy {np.mean([r['dy'] for r in rr]):5.1f} cm dyaw {np.mean([r['dyaw'] for r in rr]):5.1f} | fell {sum(r['fell'] for r in rr)}",
                  flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), default=float)
