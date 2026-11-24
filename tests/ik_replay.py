"""Offline replay of the teleop arm IK on a recorded run (2026-11-24).

Feeds the operator's wrist targets from <run>/teleop_pose.csv (one row per teleop frame, written by dwbc_latency.py)
through TeleopRetargetingIK + GlobalBodyIKSolver exactly as the teleop loop does (same robot model, waist in the IK,
posture cost, simulated clock = the frame's t_mono), and reports how smooth the IK output is. No ROS, no robot.

    python tests/ik_replay.py <run_dir> [--ik path/to/dwbc_ik_global.py] [--t0 S --t1 S] [--json out.json]
                                        [--set NAME=VALUE ...]      # override GlobalBodyIKSolver class attributes

Run one solver variant per process (dwbc_ik_global patches TeleopRetargetingIK on import).

Metrics (arm joints only, active frames only):
  jump      a frame where some arm joint of the IK output moves faster than JUMP_RAD_S while both wrist targets move
            slower than SLOW_M_S - the arm swings although the operator's hand hardly moves
  at_vmax   a frame where some arm joint sits at the solver's VMAX cap (the output was clipped)
  switches  IK solution switches (and the joint change of each), waist mode changes
  err       wrist error of the published joints against the target (cm + deg/20 per hand, both hands summed)
  match     (baseline only) how closely the replay reproduces the logged IK output - checks the harness itself
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

JUMP_RAD_S = 3.0
SLOW_M_S = 0.5


def load_solver(path):
    spec = importlib.util.spec_from_file_location("dwbc_ik_global", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dwbc_ik_global"] = m
    spec.loader.exec_module(m)
    return m


def build(ik_mod, posture_cost):
    from decoupled_wbc.control.teleop.solver.body import body_ik_solver_settings as BS
    import decoupled_wbc.control.teleop.teleop_retargeting_ik as TRI
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    init = BS.BodyIKSolverSettings.__init__

    def tuned(self):
        init(self)
        self.posture_cost = posture_cost

    BS.BodyIKSolverSettings.__init__ = tuned
    TRI.BodyIKSolver = ik_mod.GlobalBodyIKSolver
    robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body", high_elbow_pose=False)
    ik = TRI.TeleopRetargetingIK(robot_model=robot, left_hand_ik_solver=None, right_hand_ik_solver=None,
                                 body_active_joint_groups=["upper_body"])
    return robot, ik


def pose_mat(v):
    """[x, y, z, qw, qx, qy, qz] -> 4x4"""
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = R.from_quat([v[4], v[5], v[6], v[3]]).as_matrix()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ik", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dwbc_ik_global.py"))
    ap.add_argument("--t0", type=float, default=None, help="seconds from the first row")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--posture-cost", type=float, default=float(os.environ.get("DWBC_IK_POSTURE_COST", "0.01")))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--json", default=None)
    ap.add_argument("--series", default=None, help="write per-frame npz (t, target, logged q, replay q)")
    a = ap.parse_args()

    mod = load_solver(os.path.abspath(a.ik))
    G = mod.GlobalBodyIKSolver
    for kv in a.set:
        k, v = kv.split("=", 1)
        old = getattr(G, k)
        setattr(G, k, type(old)(v) if not isinstance(old, bool) else v in ("1", "True", "true"))
    G.verbose = False
    sim = {"t": 0.0}
    G.clock = staticmethod(lambda: sim["t"])

    robot, ik = build(mod, a.posture_cost)
    full = ik.full_robot
    up = list(full.get_joint_group_indices("upper_body"))
    names = [full.joint_names[i] for i in up]
    arm = np.array([("hand" not in n and not n.startswith("waist")) for n in names])
    waist = np.array([n.startswith("waist") for n in names])
    default_up = np.asarray(full.default_body_pose, dtype=float)[up]
    lf = full.supplemental_info.hand_frame_names["left"]
    rf = full.supplemental_info.hand_frame_names["right"]
    solver = ik.body_ik_solver

    rows = list(csv.DictReader(open(os.path.join(a.run, "teleop_pose.csv"))))
    t_all = np.array([int(r["t_mono_ns"]) for r in rows]) / 1e9
    tb = t_all[0]
    W = [f"wrist_l{i}" for i in range(7)] + [f"wrist_r{i}" for i in range(7)]
    Q = [f"q{i}" for i in range(31)]

    out_t, out_tgt, out_log, out_rep, out_err, out_new = [], [], [], [], [], []
    prev_active, prev_log = False, None
    n_reset = 0
    pub_cfg = None
    diag, q_act = [], []
    for r in rows:
        t = int(r["t_mono_ns"]) / 1e9
        if (a.t0 is not None and t - tb < a.t0) or (a.t1 is not None and t - tb > a.t1):
            continue
        active = r["active"] in ("1", "True")
        try:
            w = np.array([float(r[k]) for k in W])
            ql = np.array([float(r[k]) for k in Q])
        except (TypeError, ValueError):
            continue
        if active and not prev_active and prev_log is not None:
            # the real loop resets the IK after an A+B+X+Y return to the standing pose: the logged output then sits
            # at the default pose when following starts again
            if float(np.abs((prev_log - default_up)[arm | waist]).max()) < 0.02:
                ik.reset()
                n_reset += 1
        prev_active, prev_log = active, ql
        if not active:
            continue
        sim["t"] = t
        body = {lf: pose_mat(w[:7]), rf: pose_mat(w[7:])}
        ik.set_goal({"body_data": body, "left_hand_data": None, "right_hand_data": None})
        q = np.asarray(ik.get_action(), dtype=float)
        out_new.append(not prev_active)
        out_t.append(t - tb)
        out_tgt.append(w)
        out_log.append(ql)
        out_rep.append(q)
        # error of the PUBLISHED joints (q_out, after blend / VMAX) against this frame's target
        if pub_cfg is None:
            import pink
            pub_cfg = pink.Configuration(solver.configuration.model, solver.configuration.model.createData(),
                                         solver.q_out.copy())
        pub_cfg.update(solver.q_out.copy())
        out_err.append(solver._task_err(pub_cfg))
        if a.series:   # per-frame diagnostics of the active solution
            cfg = solver.configuration
            sv = []
            for f in (lf, rf):
                J = solver.tasks[f].compute_jacobian(cfg)
                sv.append(float(np.linalg.svd(J[:, solver.armwaist & ~solver.is_waist], compute_uv=False)[-1]))
            diag.append([{"locked": 0, "free": 1, "returning": 2}.get(getattr(solver, "waist_mode", ""), -1),
                         float(solver.blend is not None), sv[0], sv[1], solver.switches,
                         float(np.abs(cfg.q - solver.q_out).max())])
            q_act.append(np.asarray(cfg.q, dtype=float).copy())

    t = np.array(out_t)
    tgt = np.array(out_tgt)
    ql = np.array(out_log)
    qr = np.array(out_rep)
    res = {"run": os.path.basename(a.run.rstrip("/")), "ik": a.ik, "set": a.set, "frames": len(t),
           "active_s": round(float(len(t)) / 20.0, 1), "resets": n_reset}

    def smooth(q):
        dt = np.diff(t)
        ok = (dt > 0.02) & (dt < 0.15)
        v = np.abs(np.diff(q[:, arm], axis=0)) / np.maximum(dt, 1e-3)[:, None]
        hv = np.maximum(np.linalg.norm(np.diff(tgt[:, :3], axis=0), axis=1),
                        np.linalg.norm(np.diff(tgt[:, 7:10], axis=0), axis=1)) / np.maximum(dt, 1e-3)
        vmax = v.max(axis=1)
        jump = ok & (vmax > JUMP_RAD_S) & (hv < SLOW_M_S)
        capped = ok & (vmax >= 0.95 * G.VMAX)
        # jerk index: mean |second difference| of the arm joints (rad/frame^2), lower = smoother
        acc = np.abs(np.diff(q[:, arm], n=2, axis=0)).mean()
        # what the control loop sees: its interpolation caps arm joints at 3.5 rad/s (a "stall" when a frame asks more)
        cap35 = ok & (vmax > 3.5)
        # jerk events: some arm joint changes speed by more than 2 rad/s from one frame to the next (> 40 rad/s^2)
        vs = np.diff(q[:, arm], axis=0) / np.maximum(dt, 1e-3)[:, None]
        jerk = ok[1:] & ok[:-1] & (np.abs(np.diff(vs, axis=0)).max(axis=1) > 2.0)
        return {"jump": int(jump.sum()), "cap35": int(cap35.sum()), "jerk_ev": int(jerk.sum()), "jump_per_min": round(60 * jump.sum() / max(1, ok.sum() / 20), 2),
                "at_vmax": int(capped.sum()), "at_vmax_pct": round(100 * capped.sum() / max(1, ok.sum()), 2),
                "acc_mrad": round(1000 * float(acc), 3)}

    res["replay"] = smooth(qr)
    res["logged"] = smooth(ql)
    d = np.abs(qr - ql)[:, arm | waist]
    res["match_deg"] = {"p50": round(float(np.degrees(np.median(d.max(axis=1)))), 2),
                        "p95": round(float(np.degrees(np.percentile(d.max(axis=1), 95))), 2)}
    se = np.array([x for x in out_err if np.isfinite(x)])
    if len(se):
        res["task_err"] = {"mean": round(float(se.mean()), 3), "p50": round(float(np.median(se)), 2), "p95": round(float(np.percentile(se, 95)), 2),
                           "frac_gt10": round(float((se > 10).mean()), 3)}
    sw = getattr(solver, "switch_log", [])
    res["switches"] = len(sw)
    res["switch_deg_p50"] = round(float(np.degrees(np.median([s[1] for s in sw]))), 1) if sw else None
    res["waist_changes"] = len(getattr(solver, "waist_log", []))
    extra = getattr(solver, "stats", None)
    if extra:
        res["stats"] = extra
    print(json.dumps(res, ensure_ascii=False))
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
    if a.series:
        np.savez_compressed(a.series, t=t, tgt=tgt, q_log=ql, q_rep=qr, names=np.array(names), diag=np.array(diag),
                            q_act=np.array(q_act), red_names=np.array(solver.robot.joint_names))


if __name__ == "__main__":
    main()
