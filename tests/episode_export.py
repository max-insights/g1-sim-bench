"""Episode export for analysis (2026-11-27): one recorded episode (g1_data/<dataset>/ep_NNNNNN + its DWBC run dir) ->
an analysis-ready, self-describing folder that a person or an LLM agent queries with pandas / DuckDB instead of
reading raw binaries.

    python tests/episode_export.py EPISODE_DIR OUT_DIR [--no-1khz]

Layer 1 (l1/, Parquet + zstd, every stream complete, columns named by joint):
    lowstate_1khz / lowstate_50hz   motor state: q / dq / tau / temp (casing, winding) / motor_state per joint, IMU
    lowcmd                          motor command: q / dq / kp / kd / tau per joint
    dex1_{l,r}_{state,cmd}          grippers
    teleop                          operator input + IK (teleop_pose.csv): PICO poses, triggers, wrist targets, nav,
                                    height, IK output per upper-body joint (ik.<joint>)
    track                           commanded vs measured arm joints at the control loop (lat_track.csv)
    lat_teleop / lat_control        per-frame timing of the teleop loop / control loop (ms)
    thermal                         motor temperatures (thermal.csv)
  Every table has t = seconds from the episode start and t_mono = CLOCK_MONOTONIC seconds (all streams aligned).
Layer 2 (l2/):
    summary.json   episode meta, per-stream rate / rows, the run scorecard of the window (dwbc_score), activity time
    events.parquet / events.jsonl   detected events with time, kind, size and the joints involved
CATALOG.md         the data dictionary (columns, units, frames, rates, caveats), generated from what was written.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dwbc_score as S        # noqa: E402

UPPER = ["waist_yaw", "waist_roll", "waist_pitch"] + [f"{s}_{j}" for s in ("left",) for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")] + [
    f"left_hand_{j}" for j in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")] + [
    f"right_{j}" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
                           "wrist_yaw")] + [
    f"right_hand_{j}" for j in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")]
ARM = [f"{s}_{j}" for s in ("left", "right") for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
OUTWARD = [("left_shoulder_roll", 1), ("left_shoulder_yaw", 1), ("right_shoulder_roll", -1), ("right_shoulder_yaw", -1)]
DOC = {}      # file -> {"rate", "rows", "cols": [...], "about"}


def write(out, name, cols, about):
    tbl = pa.table(cols)
    path = os.path.join(out, "l1", name + ".parquet")
    pq.write_table(tbl, path, compression="zstd")
    t = np.asarray(cols["t"])
    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else None
    DOC[f"l1/{name}.parquet"] = dict(rows=len(t), rate_hz=None if rate is None else round(rate, 1),
                                     mb=round(os.path.getsize(path) / 1e6, 2), cols=list(cols), about=about)


def stream(ep, name):
    p = os.path.join(ep, "streams", name)
    if not os.path.exists(p + ".bin"):
        return None
    dt = json.load(open(p + ".dtype.json"))
    d = np.dtype({"names": dt["dtype"]["names"], "formats": dt["dtype"]["formats"]})
    return np.fromfile(p + ".bin", dtype=d)


def window(t_mono, t0, t1):
    return (t_mono >= t0) & (t_mono <= t1)


def flat(a, joints, t0, fields):
    cols = {"t": a["t_mono_ns"].astype(np.float64) / 1e9 - t0, "t_mono": a["t_mono_ns"].astype(np.float64) / 1e9}
    for f in fields:
        v = a[f]
        if v.ndim == 1:
            cols[f] = v
        elif v.ndim == 2:
            names = joints if v.shape[1] == len(joints) else [str(i) for i in range(v.shape[1])]
            for i, n in enumerate(names):
                cols[f"{f}.{n}"] = v[:, i]
        else:   # temp [29, 2] = casing / winding
            for i, n in enumerate(joints):
                cols[f"{f}_casing.{n}"] = v[:, i, 0]
                cols[f"{f}_winding.{n}"] = v[:, i, 1]
    return cols


def csv_table(path, t0, t1, tcol, scale=1.0):
    d = S.read_csv(path)
    if d is None or tcol not in d:
        return None
    t = d[tcol].astype(np.float64) * scale
    m = window(t, t0, t1)
    cols = {"t": t[m] - t0, "t_mono": t[m]}
    for k, v in d.items():
        if k != tcol:
            cols[k] = np.asarray(v)[m]
    return cols


def events_arm(t, q, tg, names, source):
    """swings > 25 deg in 0.5 s with both hands < 5 cm, outward shoulder > 15 deg with hands < 6 cm, as dwbc_score."""
    out, k, n = [], 0, len(t)
    ix = {nm: i for i, nm in enumerate(names)}
    while k < n:
        j = k
        while j + 1 < n and t[j + 1] - t[k] <= 0.5 and t[j + 1] - t[j] < 0.15:
            j += 1
        if j > k:
            dh = max(np.linalg.norm(tg[j, :3] - tg[k, :3]), np.linalg.norm(tg[j, 7:10] - tg[k, 7:10]))
            dq = q[j] - q[k]
            sw = float(np.degrees(np.abs(dq).max()))
            ow = max(sg * dq[ix[nm]] for nm, sg in OUTWARD)
            hit = None
            if dh < 0.05 and sw > 25:
                hit = ("swing25", sw)
            elif dh < 0.06 and math.degrees(ow) > 15:
                hit = ("outward15", math.degrees(ow))
            if hit:
                top = np.argsort(-np.abs(dq))[:4]
                out.append(dict(t=round(float(t[k]), 3), dur=round(float(t[j] - t[k]), 3), kind=hit[0], source=source,
                                size=round(hit[1], 1), hand_move_cm=round(100 * float(dh), 1),
                                joints={names[i]: round(float(np.degrees(dq[i])), 1) for i in top}))
                k = j
                continue
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--no-1khz", action="store_true")
    a = ap.parse_args()
    ep = os.path.realpath(a.episode)
    meta = json.load(open(os.path.join(ep, "meta.json")))
    run = (meta.get("dwbc") or {}).get("run_dir") or next(iter(meta.get("dwbc_run_dirs") or []), None)
    pose_all = S.read_csv(os.path.join(run, "teleop_pose.csv"), {"t_wall_ns", "t_mono_ns"})
    off = S.wall_to_mono(pose_all)
    t0, t1 = meta["t_start_ns"] / 1e9 + off, meta["t_stop_ns"] / 1e9 + off
    os.makedirs(os.path.join(a.out, "l1"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "l2"), exist_ok=True)
    joints = [n.replace("_joint", "") for n in meta["joint_names"]]

    ls = stream(ep, "lowstate")
    ls = ls[window(ls["t_mono_ns"] / 1e9, t0, t1)]
    f = ["tick", "mode_machine", "q", "dq", "tau", "temp", "motor_state", "imu_quat", "imu_rpy", "imu_gyro", "imu_acc"]
    about = ("motor state as received (unitree lowstate, ~1 kHz): q rad, dq rad/s, tau Nm (estimated), temp_casing / "
             "temp_winding degC, motor_state bit flags (0 = ok), imu_quat [w,x,y,z] (order to be confirmed on the robot), "
             "imu_rpy rad, imu_gyro rad/s, imu_acc m/s^2 (pelvis IMU). Joint offsets in meta.joint_offsets.")
    if not a.no_1khz:
        write(a.out, "lowstate_1khz", flat(ls, joints, t0, f), about)
    write(a.out, "lowstate_50hz", flat(ls[::20], joints, t0, f), about.replace("~1 kHz", "every 20th sample, ~50 Hz"))
    lc = stream(ep, "lowcmd")
    if lc is not None:
        lc = lc[window(lc["t_mono_ns"] / 1e9, t0, t1)]
        write(a.out, "lowcmd", flat(lc, joints, t0, ["mode_machine", "q", "dq", "kp", "kd", "tau"]),
              "motor command sent to the robot: target q rad, dq rad/s, kp, kd, feed-forward tau Nm")
    for side in ("l", "r"):
        for kind in ("state", "cmd"):
            s = stream(ep, f"dex1_{side}_{kind}")
            if s is not None:
                s = s[window(s["t_mono_ns"] / 1e9, t0, t1)]
                write(a.out, f"dex1_{side}_{kind}", flat(s, [], t0, [x for x in s.dtype.names if x not in ("t_recv_ns", "t_mono_ns")]),
                      f"{'left' if side == 'l' else 'right'} Dex1 gripper {kind}: q (open {meta.get('gripper_calib', {}).get('left' if side == 'l' else 'right', {}).get('q_open')}, closed ~0)")

    tp = csv_table(os.path.join(run, "teleop_pose.csv"), t0, t1, "t_mono_ns", 1e-9)
    if tp is not None:
        ren = {}
        for i, n in enumerate(UPPER):
            ren[f"q{i}"] = f"ik.{n}"
        for i, n in enumerate(["x", "y", "z", "qw", "qx", "qy", "qz"]):
            ren[f"wrist_l{i}"], ren[f"wrist_r{i}"] = f"wrist_target_l.{n}", f"wrist_target_r.{n}"
        for p in ("left_pose", "right_pose", "head_pose"):
            for i, n in enumerate(["x", "y", "z", "qx", "qy", "qz", "qw"]):
                ren[f"{p}{i}"] = f"pico_{p}.{n}"
        tp = {ren.get(k, k): v for k, v in tp.items()}
        write(a.out, "teleop", tp, "teleop loop, one row per published goal (~20 Hz): pico_* = raw XRoboToolkit poses in the "
              "headset world frame [x,y,z,qx,qy,qz,qw]; *_trigger / *_grip 0..1; wrist_target_* = IK targets in the robot "
              "frame [x,y,z,qw,qx,qy,qz] m; nav = [vx, vy, wz]; height = base height command m; ik.<joint> = IK output "
              "(target upper-body pose) rad; active = teleop engaged")
    tr = csv_table(os.path.join(run, "lat_track.csv"), t0, t1, "t")
    if tr is not None:
        tr = {({f"cmd{i}": f"cmd.{n}" for i, n in enumerate(ARM)} | {f"meas{i}": f"meas.{n}" for i, n in enumerate(ARM)}).get(k, k): v
              for k, v in tr.items()}
        write(a.out, "track", tr, "arm joints at the control loop: cmd = target sent (after interpolation / speed limits) rad, "
              "meas = measured rad")
    for nm, tcol in (("lat_teleop", "t_read"), ("lat_control", "t_read")):
        x = csv_table(os.path.join(run, nm + ".csv"), t0, t1, tcol)
        if x is not None:
            write(a.out, nm, x, "per-frame timing (ms): period_ms between frames, streamer_ms PICO read, ik_ms arm IK, "
                  "action_ms / pub_ms publish; pico_new 0 = the headset sent no new frame; *_ns PICO / wall clocks"
                  + ("; t_pub..t_sent = monotonic timestamps of each hop to the robot" if nm == "lat_control" else ""))
    thp = os.path.join(run, "thermal.csv")
    th = None
    if os.path.exists(thp):
        import pandas as pd
        th = pd.read_csv(thp)
    if th is not None and "unix_t" in th:
        tm = th["unix_t"].to_numpy(np.float64) + off
        m = window(tm, t0, t1)
        write(a.out, "thermal", {"t": tm[m] - t0, "t_mono": tm[m],
                                 **{k: th[k].to_numpy()[m] for k in th.columns if k != "unix_t"}},
              "thermal monitor (~1 Hz): casing / winding degC, tau Nm, q rad per leg / waist motor")

    # ---------------------------------------------------------------- layer 2
    ev = []
    if tp is not None:
        t = tp["t"]
        q = np.stack([tp[f"ik.{n}"] for n in ARM], 1)
        tg = np.stack([tp[f"wrist_target_l.{c}"] for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                      + [tp[f"wrist_target_r.{c}"] for c in ("x", "y", "z", "qw", "qx", "qy", "qz")], 1)
        ev += events_arm(t, q, tg, ARM, "ik_output")
        ls50 = ls[::20]
        tl = ls50["t_mono_ns"] / 1e9 - t0
        qm = np.stack([ls50["q"][:, joints.index(n)] for n in ARM], 1)
        tgm = np.stack([np.interp(tl, t, tg[:, i]) for i in range(14)], 1)
        ev += events_arm(tl, qm, tgm, ARM, "motor_measured")
        w = np.stack([tp[f"ik.{n}"] for n in ("waist_yaw", "waist_roll", "waist_pitch")], 1)
        k = 0
        while k < len(t):
            j = int(np.searchsorted(t, t[k] + 0.5))
            if j < len(t) and np.degrees(np.abs(w[j] - w[k]).max()) > 5:
                ev.append(dict(t=round(float(t[k]), 3), dur=round(float(t[j] - t[k]), 3), kind="waist5", source="ik_output",
                               size=round(float(np.degrees(np.abs(w[j] - w[k]).max())), 1), hand_move_cm=None,
                               joints={n: round(float(np.degrees(w[j, i] - w[k, i])), 1) for i, n in enumerate(("waist_yaw", "waist_roll", "waist_pitch"))}))
                k = j
            else:
                k += 1
    lt = S.read_csv(os.path.join(run, "lat_teleop.csv"))
    if lt is not None:
        tt = lt["t_read"].astype(np.float64)
        m = window(tt, t0, t1)
        tt, pn = tt[m] - t0, np.asarray(lt["pico_new"])[m]
        k = 0
        while k < len(tt):
            if pn[k] == 0:
                j = k
                while j + 1 < len(tt) and pn[j + 1] == 0:
                    j += 1
                if tt[j] - tt[k] >= 0.15:
                    ev.append(dict(t=round(float(tt[k]), 3), dur=round(float(tt[j] - tt[k]), 3), kind="headset_gap",
                                   source="lat_teleop", size=round(float(tt[j] - tt[k]) * 1000), hand_move_cm=None, joints={}))
                k = j + 1
            else:
                k += 1
    for ts, ti, direction, extra in S.step_events(run) or []:     # (t_start, t_idle, direction, extra steps)
        if t0 <= ts <= t1:
            ev.append(dict(t=round(ts - t0, 3), dur=round(ti - ts, 3) if ti else None, kind="step", source="control.log",
                           size=extra, hand_move_cm=None, joints={"direction": direction}))
    ev.sort(key=lambda e: e["t"])
    with open(os.path.join(a.out, "l2", "events.jsonl"), "w") as fh:
        for e in ev:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if ev:
        pq.write_table(pa.table({"t": [e["t"] for e in ev], "dur": [e["dur"] for e in ev], "kind": [e["kind"] for e in ev],
                                 "source": [e["source"] for e in ev], "size": [None if e["size"] is None else float(e["size"]) for e in ev],
                                 "hand_move_cm": [e["hand_move_cm"] for e in ev],
                                 "joints": [json.dumps(e["joints"], ensure_ascii=False) for e in ev]}),
                       os.path.join(a.out, "l2", "events.parquet"), compression="zstd")
    counts = {}
    for e in ev:
        counts[e["kind"] + "/" + e["source"]] = counts.get(e["kind"] + "/" + e["source"], 0) + 1
    try:
        card = S.score_episode(ep)
    except Exception as e:  # noqa: BLE001
        card = {"error": str(e)[:300]}
    act = {}
    if tp is not None:
        el = np.maximum(np.abs(tp["ik.left_elbow"]), np.abs(tp["ik.right_elbow"]))
        dtp = np.diff(tp["t"], append=tp["t"][-1])
        act["arm_near_straight_s (IK elbow > 1.2 rad or < 0.25 rad)"] = round(float(dtp[(el > 1.2) | (el < 0.25)].sum()), 1)
        act["teleop_active_s"] = round(float(dtp[np.asarray(tp["active"]) > 0].sum()), 1)
        if "height" in tp:
            h = np.asarray(tp["height"], dtype=float)
            act["squat_s (height < 0.69 m)"] = round(float(dtp[np.nan_to_num(h, nan=0.74) < 0.69].sum()), 1)
        nav = np.stack([np.nan_to_num(np.asarray(tp[f"nav{i}"], dtype=float)) for i in range(3)], 1)
        act["walking_cmd_s (|nav| > 0.05)"] = round(float(dtp[np.linalg.norm(nav, axis=1) > 0.05].sum()), 1)
    summary = {"episode": f"{meta.get('dataset')}/{meta.get('episode_id')}", "task": meta.get("task"), "operator": meta.get("operator"),
               "duration_s": meta.get("duration_s"), "run_dir": os.path.basename(run), "controller": meta.get("controller"),
               "window_mono": [round(t0, 3), round(t1, 3)], "joint_names": joints, "joint_offsets": meta.get("joint_offsets"),
               "cams": meta.get("cams"), "activity": act, "event_counts": counts, "scorecard": card,
               "files": {k: {kk: v[kk] for kk in ("rows", "rate_hz", "mb")} for k, v in DOC.items()}}
    json.dump(summary, open(os.path.join(a.out, "l2", "summary.json"), "w"), ensure_ascii=False, indent=1, default=float)

    lines = [f"# {summary['episode']} — {meta.get('task')}", "",
             f"Duration {meta.get('duration_s')} s, run `{os.path.basename(run)}`, controller {meta.get('controller')}. "
             "All tables share `t` (s from the episode start) and `t_mono` (workstation CLOCK_MONOTONIC s).", "",
             "Start with `l2/summary.json` (activity time, event counts, the 6-dimension scorecard) and `l2/events.parquet`; "
             "then load only the window you need from `l1/` (e.g. `pd.read_parquet(p, columns=[...])` or DuckDB).", "",
             "Joint order (lowstate / lowcmd): " + ", ".join(joints), "",
             "Events (`l2/events.*`): swing25 = an arm joint moves > 25 deg within 0.5 s while both wrist targets move < 5 cm; "
             "outward15 = shoulder roll / yaw moves outward > 15 deg with the hands < 6 cm; waist5 = a waist joint moves > 5 deg "
             "within 0.5 s; headset_gap = >= 150 ms without a new PICO frame; step = single-step command (control.log). "
             "source ik_output = IK command (teleop), motor_measured = lowstate at 50 Hz.", "",
             "Caveats: imu_quat order is not yet confirmed on the robot; left_wrist_yaw has a joint offset (meta.joint_offsets); "
             "the camera video (cams in summary) is not exported — ask for frames at given times.", "", "## Files", ""]
    for k, v in DOC.items():
        cs = v["cols"]
        shown = ", ".join(cs[:14]) + (f", … ({len(cs)} columns)" if len(cs) > 14 else "")
        lines += [f"### `{k}` — {v['rows']} rows, {v['rate_hz']} Hz, {v['mb']} MB", "", v["about"], "", f"Columns: {shown}", ""]
    open(os.path.join(a.out, "CATALOG.md"), "w").write("\n".join(lines))
    print(json.dumps({"out": a.out, "files": len(DOC), "mb": round(sum(v["mb"] for v in DOC.values()), 1), "events": counts},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
