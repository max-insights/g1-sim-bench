"""Teleop arm-IK benchmark: recorded operator input (teleop_pose.csv) frozen as simulation assets, split into
train / test, replayed offline through the arm IK (tests/ik_replay.py: reproduces the logged real-robot IK output to
0 deg), scored with the same metrics every time (2026-11-25).

    python tests/ik_bench.py add  real_YYYYMMDD_HHMMSS --split train|test [--t0 S --t1 S] [--note "..."]
    python tests/ik_bench.py list
    python tests/ik_bench.py run  NAME [--split train|test|all] [--smooth 0|1] [--set ATTR=VALUE ...] [-j 4] [--skip-existing]
    python tests/ik_bench.py compare NAME [NAME ...] [--split ...] [--by-activity]

Assets live in $IK_BENCH (default /data/robot_data/g1/ik_bench): assets/<id>/teleop_pose.csv is a frozen copy (the
run dir may be cleaned up later), manifest.json holds window / split / note. Split by RUN, never by frame: frames of
one run are strongly correlated. Rule: tune on train only; test is looked at once per candidate, to confirm.
Replays run at nice 19 (safe next to a live robot run, the teleop IK shares the CPU).

Scorecard (per active minute unless noted):
  swing40 / swing25  some arm joint moves > 40 / 25 deg within 0.5 s while both wrist targets move < 5 cm
                     ("the arm jerks although the hand hardly moves" - what the operator sees at the reach limit)
  jerk               frames where an arm joint's speed changes by > 2 rad/s between frames
  err3 / err5        seconds with wrist error > 3 / 5 (cm + deg/20, both hands), 1 s after each A+X start excluded
  err_mean           mean wrist error
  roll_range         waist roll range (deg)      elbow_max  largest elbow angle (rad, straight = 1.41)
  near_limit         share of frames with an arm close to straight (smallest hand-Jacobian singular value < 0.04)
  ms                 IK time per frame
  --by-activity      the same per activity (frames labelled from the asset): squat (height command > 5 cm below
                     standing), step / turn (single-step events in the run's control.log, continuous walking or turning
                     from the nav command), reach (an arm close to straight), other. A frame can carry several labels.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("IK_BENCH", "/data/robot_data/g1/ik_bench")
LOGS = os.path.expanduser("~/piper_ws/dwbc_logs")
MAN = os.path.join(ROOT, "manifest.json")


def manifest():
    return json.load(open(MAN)) if os.path.exists(MAN) else {"assets": []}


def save(m):
    os.makedirs(ROOT, exist_ok=True)
    json.dump(m, open(MAN + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(MAN + ".tmp", MAN)


def cmd_add(a):
    src = os.path.join(LOGS, a.run, "teleop_pose.csv")
    if not os.path.exists(src):
        sys.exit(f"no {src}")
    aid = a.run.replace("real_", "") + (f"_{int(a.t0)}" if a.t0 else "")
    m = manifest()
    if any(x["id"] == aid for x in m["assets"]):
        sys.exit(f"asset {aid} exists")
    d = os.path.join(ROOT, "assets", aid)
    os.makedirs(d, exist_ok=True)
    shutil.copy2(src, d)
    ctl = os.path.join(LOGS, a.run, "control.log")
    if os.path.exists(ctl):
        shutil.copy2(ctl, d)
    m["assets"].append({"id": aid, "run": a.run, "t0": a.t0, "t1": a.t1, "split": a.split, "note": a.note})
    save(m)
    print(f"added {aid} ({a.split})")


def cmd_list(a):
    for x in manifest()["assets"]:
        print(f"{x['id']:22s} {x['split']:5s} t0={x['t0']} t1={x['t1']}  {x.get('note') or ''}")


def chosen(split):
    return [x for x in manifest()["assets"] if split == "all" or x["split"] == split]


def cmd_run(a):
    out = os.path.join(ROOT, "results", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump({"smooth": a.smooth, "set": a.set}, open(os.path.join(out, "config.json"), "w"))
    procs = []
    for x in chosen(a.split):
        if a.skip_existing and os.path.exists(os.path.join(out, x["id"] + ".npz")):
            continue
        c = ["nice", "-n", "19", sys.executable, os.path.join(HERE, "ik_replay.py"), os.path.join(ROOT, "assets", x["id"]),
             "--json", os.path.join(out, x["id"] + ".json"), "--series", os.path.join(out, x["id"] + ".npz")]
        if x["t0"] is not None:
            c += ["--t0", str(x["t0"])]
        if x["t1"] is not None:
            c += ["--t1", str(x["t1"])]
        if a.smooth is not None:
            c += ["--smooth", a.smooth]
        for s in a.set:
            c += ["--set", s]
        running = [p for p in procs if p.poll() is None]
        while len(running) >= a.j:
            running[0].wait()
            running = [p for p in procs if p.poll() is None]
        procs.append(subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=open(os.path.join(out, x["id"] + ".err"), "w")))
    for p in procs:
        p.wait()
    cmd_compare(argparse.Namespace(names=[a.name], split=a.split))


def swings(t, q, tgt, arm, deg):
    c, k, n = 0, 0, len(t)
    while k < n:
        j = k
        while j + 1 < n and t[j + 1] - t[k] <= 0.5 and t[j + 1] - t[j] < 0.15:
            j += 1
        if j > k:
            dq = np.abs(q[j, arm] - q[k, arm]).max()
            dh = max(np.linalg.norm(tgt[j, :3] - tgt[k, :3]), np.linalg.norm(tgt[j, 7:10] - tgt[k, 7:10]))
            if dq > np.radians(deg) and dh < 0.05:
                c, k = c + 1, j
                continue
        k += 1
    return c


def step_events(aid):
    """[(t_start, t_end, "step"|"turn")] from the single-step lines of control.log, on the csv clock (t_mono s)."""
    import re
    f = os.path.join(ROOT, "assets", aid, "control.log")
    if not os.path.exists(f):
        run = next(x["run"] for x in manifest()["assets"] if x["id"] == aid)
        src = os.path.join(LOGS, run, "control.log")
        if not os.path.exists(src):
            return []
        shutil.copy2(src, f)
    ev, cur = [], None
    for line in open(f, errors="replace"):
        m = re.search(r"\[dwbc_step\] ([0-9.]+) (start|idle|turn)\s*(\S*)", line)
        if not m:
            continue
        t, what, arg = float(m.group(1)), m.group(2), m.group(3)
        if what == "start" or (what == "turn" and arg):
            if cur:
                ev.append((cur[0], t, cur[1]))
            cur = (t, "turn" if (what == "turn" or arg.startswith("turn")) else "step")
        elif what == "idle" and cur:
            ev.append((cur[0], t, cur[1]))
            cur = None
    return ev


_LAB = {}


def labels(aid, t):
    """per replay frame (t = s from the csv's first row): dict label -> bool array"""
    import csv
    if aid not in _LAB:
        rows = list(csv.DictReader(open(os.path.join(ROOT, "assets", aid, "teleop_pose.csv"))))

        def col(k):
            o = np.full(len(rows), np.nan)
            for i, r in enumerate(rows):
                try:
                    o[i] = float(r[k])
                except (TypeError, ValueError, KeyError):
                    pass
            return o
        tm = np.array([int(r["t_mono_ns"]) for r in rows]) / 1e9
        _LAB[aid] = (tm, col("height"), np.stack([col(f"nav{i}") for i in range(3)], 1))
    tm, hh, nn = _LAB[aid]
    idx = np.clip(np.searchsorted(tm - tm[0], t - 1e-4), 0, len(tm) - 1)
    h, nav = hh[idx], nn[idx]
    stand = np.nanpercentile(hh, 90)
    lab = {"squat": h < stand - 0.05,
           "step": np.nan_to_num(np.abs(nav[:, :2])).max(1) > 0.05,
           "turn": np.nan_to_num(np.abs(nav[:, 2])) > 0.05}
    tabs = tm[idx]
    for a, b, k in step_events(aid):
        lab[k] |= (tabs >= a) & (tabs <= b + 0.5)
    return lab


def score(name, ids, activity=None):
    tot = {"min": 0.0, "swing40": 0, "swing25": 0, "jerk": 0, "err3": 0.0, "err5": 0.0, "near": 0.0, "ms": []}
    errs, roll, elbow = [], 0.0, 0.0
    for aid in ids:
        f = os.path.join(ROOT, "results", name, aid + ".npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        t, n, tgt, e, q, d = z["t"], list(z["names"]), z["tgt"], z["err"], z["q_rep"], z["diag"]
        if activity is not None:   # only the frames with this label (gaps between them are skipped by the dt filter)
            lab = labels(aid, t)
            sv = np.minimum(d[:, 2], d[:, 3])
            if activity == "reach":
                keep = sv < 0.04
            elif activity in lab:
                keep = lab[activity]
            else:
                keep = ~(lab["squat"] | lab["step"] | lab["turn"] | (sv < 0.04))
            if keep.sum() < 2:
                continue
            t, tgt, e, q, d = t[keep], tgt[keep], e[keep], q[keep], d[keep]
        arm = np.array([("hand" not in x and not x.startswith("waist")) for x in n])
        dt = np.diff(t)
        ok = (dt > 0.02) & (dt < 0.15)
        mins = len(t) / 1200.0
        vs = np.diff(q[:, arm], axis=0) / np.maximum(dt, 1e-3)[:, None]
        tot["jerk"] += int((ok[1:] & ok[:-1] & (np.abs(np.diff(vs, axis=0)).max(1) > 2.0)).sum())
        tot["swing40"] += swings(t, q, tgt, arm, 40)
        tot["swing25"] += swings(t, q, tgt, arm, 25)
        w = np.ones(len(t), bool)
        for s in [t[0]] + [t[k] for k in range(1, len(t)) if t[k] - t[k - 1] > 0.5]:
            w &= ~((t >= s) & (t < s + 1.0))
        errs.append(e[w])
        tot["err3"] += 0.05 * float((e[w] > 3).sum())
        tot["err5"] += 0.05 * float((e[w] > 5).sum())
        tot["near"] += float((np.minimum(d[:, 2], d[:, 3]) < 0.04).sum()) / 1200.0
        tot["min"] += mins
        roll = max(roll, float(np.degrees(np.ptp(q[:, n.index("waist_roll_joint")]))))
        elbow = max(elbow, float(q[:, [n.index("left_elbow_joint"), n.index("right_elbow_joint")]].max()))
        tot["ms"].append(json.load(open(os.path.join(ROOT, "results", name, aid + ".json")))["ms_per_frame"])
    m = tot["min"]
    if m <= 0:
        return {"min": 0.0}
    e = np.concatenate(errs)
    return {"min": round(m, 1), "swing40": round(tot["swing40"] / m, 2), "swing25": round(tot["swing25"] / m, 2),
            "jerk": round(tot["jerk"] / m, 2), "err3": round(tot["err3"] / m, 2), "err5": round(tot["err5"] / m, 2),
            "err_mean": round(float(np.nanmean(e)), 2), "roll_range": round(roll, 1), "elbow_max": round(elbow, 2),
            "near_limit": round(100 * tot["near"] / m, 1), "ms": round(float(np.mean(tot["ms"])), 1)}


def cmd_compare(a):
    ids = [x["id"] for x in chosen(a.split)]
    if getattr(a, "by_activity", False):
        keys = ["swing40", "swing25", "jerk", "err3", "err5", "err_mean"]
        print(f"split {a.split}: {len(ids)} assets, by activity (per minute of that activity)")
        print(f"{'activity':8s} {'variant':16s} {'min':>5s} " + " ".join(f"{k:>9s}" for k in keys))
        for act in ("squat", "step", "turn", "reach", "other"):
            for name in a.names:
                sc = score(name, ids, act)
                if sc is None or not sc.get("min"):
                    continue
                print(f"{act:8s} {name:16s} {sc['min']:5.1f} " + " ".join(f"{sc[k]:9}" for k in keys))
            print()
        return
    keys = ["swing40", "swing25", "jerk", "err3", "err5", "err_mean", "roll_range", "elbow_max", "near_limit", "ms"]
    print(f"split {a.split}: {len(ids)} assets")
    print(f"{'variant':16s} {'min':>5s} " + " ".join(f"{k:>10s}" for k in keys))
    for name in a.names:
        s = score(name, ids)
        if s is None:
            print(f"{name:16s} (not run on every asset of this split)")
            continue
        json.dump(s, open(os.path.join(ROOT, "results", name, f"score_{a.split}.json"), "w"))
        print(f"{name:16s} {s['min']:5.1f} " + " ".join(f"{s[k]:10}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("run")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--t1", type=float, default=None)
    p.add_argument("--note", default="")
    sub.add_parser("list")
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--split", default="train", choices=["train", "test", "all"])
    p.add_argument("--smooth", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("-j", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true", help="keep results of assets already replayed")
    p = sub.add_parser("compare")
    p.add_argument("names", nargs="+")
    p.add_argument("--by-activity", action="store_true")
    p.add_argument("--split", default="train", choices=["train", "test", "all"])
    a = ap.parse_args()
    {"add": cmd_add, "list": cmd_list, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
