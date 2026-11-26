"""Simulation scorecard: one command that replays / simulates a candidate configuration and scores it with the same
dimensions and anchors as the real-robot scorecard (dwbc_score.py), plus a regression gate against a baseline
(2026-11-26). This is what a person or an agent calls to try a change before it goes near the robot.

    python tests/sim_eval.py ik   NAME [--split train|test] [--asof YYYYMMDD] [--smooth 0|1] [--set ATTR=V ...] [--ik FILE] [-j 4]
    python tests/sim_eval.py walk NAME [--params '{"waist_kd":[5,5,10]}'] [--dirs back,left,...] [-j 5]
    python tests/sim_eval.py show NAME [--split ...]                 # the stored scorecard
    python tests/sim_eval.py gate BASE CAND [--split ...] [--target jumps,...]   # pass / fail + reasons (exit 0 / 1)
    python tests/sim_eval.py events NAME [--kind swing|outward|err|jerk] [--top 12] [--asof ...]   # what happened at the worst events

ik   arm IK replay of the frozen teleop recordings (tests/ik_bench.py assets; reproduces the real IK output to 0 deg).
     Dimensions (sim): 突变 jumps (jerk / swing>25 / outward>15 per minute on the IK output, same anchors as real),
     抖动 jitter (IK output while the hands hold still), 手跟随 track (seconds per minute with the hand > 3 cm off
     target, mean error), 腰活动 waist (direction reversals > 1 deg and travel per minute), 逆解 ik (ms per frame).
     --asof YYYYMMDD keeps only assets recorded before that day (backtests: what was known then).
walk single steps of the leg policy in MuJoCo (tests/walk_sim.py, 1 control tick of actuation delay = real-like).
     Dimension 走路流畅度 walking (pelvis roll / roll rate / pitch per step, extra steps, same anchors as real) + falls.

Results: $IK_BENCH/results/simeval/<NAME>/ (ik: per-asset json / npz + score_<split>.json; walk: walk.json + score.json).
Everything runs at nice 19 and refuses to start while the real chain runs (pass --force to override).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import ik_bench as B          # noqa: E402
import dwbc_score as S        # noqa: E402

OUT = os.path.join(B.ROOT, "results", "simeval")
SIM_ANCHORS = {               # sim-only dimensions (value at 100, value at 0)
    "err3_s_min": (0.0, 2.0),
    "err_mean": (0.05, 0.6),
    "waist_rev_min": (5.0, 40.0),
    "waist_travel_dpm": (50.0, 400.0),
}
DIMS = [("jumps", "突变"), ("jitter", "抖动"), ("track", "手跟随"), ("waist", "腰活动"), ("ik", "逆解"),
        ("walking", "走路流畅度")]
GATE_DROP = 5        # a non-target dimension may lose at most this many points
GATE_GAIN = 5        # each target dimension must gain at least this many points


def lin(v, key):
    a = SIM_ANCHORS.get(key) or S.ANCHORS[key]
    good, bad = a
    if v is None or not np.isfinite(v):
        return None
    return float(np.clip(100.0 * (bad - v) / (bad - good), 0.0, 100.0))


def real_chain_running():
    r = subprocess.run(["pgrep", "-f", "[d]wbc_run\\.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


# ------------------------------------------------------------------------------------------------ ik
def assets(split, asof):
    out = []
    for x in B.chosen(split):
        day = x["id"][:8]
        if asof and day >= asof:
            continue
        out.append(x)
    return out


def run_ik(a):
    d = os.path.join(OUT, a.name)
    os.makedirs(d, exist_ok=True)
    json.dump({"kind": "ik", "smooth": a.smooth, "set": a.set, "asof": a.asof, "ik": a.ik},
              open(os.path.join(d, "config.json"), "w"))
    procs = []
    xs = assets(a.split, a.asof)
    for x in xs:
        js = os.path.join(d, x["id"] + ".json")
        if os.path.exists(js) and os.path.exists(os.path.join(d, x["id"] + ".npz")):
            continue                                            # same name = same config: reuse
        c = ["nice", "-n", "19", sys.executable, os.path.join(HERE, "ik_replay.py"), os.path.join(B.ROOT, "assets", x["id"]),
             "--json", js, "--series", os.path.join(d, x["id"] + ".npz")]
        if x["t0"] is not None:
            c += ["--t0", str(x["t0"])]
        if x["t1"] is not None:
            c += ["--t1", str(x["t1"])]
        if a.smooth is not None:
            c += ["--smooth", a.smooth]
        if a.ik:
            c += ["--ik", a.ik]
        for s in a.set:
            c += ["--set", s]
        while len([p for p in procs if p.poll() is None]) >= a.j:
            next(p for p in procs if p.poll() is None).wait()
        procs.append(subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, x["id"] + ".err"), "w")))
    bad = [p for p in procs if p.wait() != 0]
    if bad:
        sys.exit(f"{len(bad)} replay(s) failed: see {d}/*.err")
    sc = score_ik(a.name, a.split, a.asof)
    print(json.dumps(sc, ensure_ascii=False, indent=1))


def waist_activity(t, qw):
    """direction reversals of more than 1 deg, and travel, summed over the 3 waist joints (deg)."""
    rev, travel = 0, 0.0
    for j in range(qw.shape[1]):
        x = np.degrees(qw[:, j])
        travel += float(np.abs(np.diff(x)).sum())
        ext, dirn = x[0], 0
        for v in x[1:]:
            if dirn >= 0 and v < ext - 1.0:
                rev += dirn > 0
                dirn, ext = -1, v
            elif dirn <= 0 and v > ext + 1.0:
                rev += dirn < 0
                dirn, ext = 1, v
            elif (dirn >= 0 and v > ext) or (dirn <= 0 and v < ext):
                ext = v
    return rev, travel


def score_ik(name, split, asof):
    d = os.path.join(OUT, name)
    tot = dict(mins=0.0, jerk=0, sw=0, ow=0, err3=0.0, rev=0, travel=0.0)
    errs, holds, ms = [], [], []
    for x in assets(split, asof):
        f = os.path.join(d, x["id"] + ".npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        t, n, tgt, e, q = z["t"], [str(v) for v in z["names"]], z["tgt"], z["err"], z["q_rep"]
        arm = [i for i, v in enumerate(n) if "hand" not in v and not v.startswith("waist")]
        wst = [i for i, v in enumerate(n) if v.startswith("waist")]
        qa = q[:, arm]
        mins = len(t) / 1200.0
        dt = np.diff(t)
        ok = (dt > 0.02) & (dt < 0.15)
        vs = np.diff(qa, axis=0) / np.maximum(dt, 1e-3)[:, None]
        tot["jerk"] += int((ok[1:] & ok[:-1] & (np.abs(np.diff(vs, axis=0)).max(1) > 2.0)).sum())
        tot["sw"] += S.swings(t, qa, tgt, 25, list(range(len(arm))))
        tot["ow"] += S.outward(t, qa, tgt)        # arm order = lat_track order (left 7, right 7)
        w = np.ones(len(t), bool)
        for s0 in [t[0]] + [t[k] for k in range(1, len(t)) if t[k] - t[k - 1] > 0.5]:
            w &= ~((t >= s0) & (t < s0 + 1.0))
        errs.append(e[w])
        tot["err3"] += 0.05 * float((e[w] > 3).sum())
        r, tr = waist_activity(t, q[:, wst])
        tot["rev"] += r
        tot["travel"] += tr
        k = 0
        while k < len(t):                       # IK-output jitter while the hands hold still (0.5 s windows)
            j = np.searchsorted(t, t[k] + 0.5)
            if j - k >= 6 and np.all(np.diff(t[k:j]) < 0.15):
                dh = max(np.ptp(tgt[k:j, 0:3], axis=0).max(), np.ptp(tgt[k:j, 7:10], axis=0).max())
                if dh < 0.015:
                    holds.append(np.degrees(np.std(qa[k:j], axis=0).max()))
            k = j
        js = json.load(open(os.path.join(d, x["id"] + ".json")))
        if js.get("ms_p50_p90_p99_max"):
            ms.append(js["ms_p50_p90_p99_max"])
        tot["mins"] += mins
    m = tot["mins"]
    if m <= 0:
        return None
    e = np.concatenate(errs)
    v = dict(jerk=tot["jerk"] / m, swing25=tot["sw"] / m, outward=tot["ow"] / m, err3=tot["err3"] / m,
             err_mean=float(np.nanmean(e)), waist_rev=tot["rev"] / m, waist_travel=tot["travel"] / m,
             hold=float(np.median(holds)) if holds else None,
             ik_p50=float(np.median([x[0] for x in ms])) if ms else None,
             ik_p99=float(np.max([x[2] for x in ms])) if ms else None)
    dims = {
        "jumps": dict(score=S.comb(lin(v["jerk"], "jerk_min"), lin(v["swing25"], "swing_min"), lin(v["outward"], "outward_min")),
                      value=round(v["jerk"], 2), detail={"大摆动>25°/分钟": round(v["swing25"], 2), "往外甩>15°/分钟": round(v["outward"], 2)}),
        "jitter": dict(score=S.comb(lin(v["hold"], "hold_std_deg")), value=None if v["hold"] is None else round(v["hold"], 3), detail={}),
        "track": dict(score=S.comb(lin(v["err3"], "err3_s_min"), lin(v["err_mean"], "err_mean")),
                      value=round(v["err3"], 3), detail={"平均偏离(cm+deg/20)": round(v["err_mean"], 3)}),
        "waist": dict(score=S.comb(lin(v["waist_rev"], "waist_rev_min"), lin(v["waist_travel"], "waist_travel_dpm")),
                      value=round(v["waist_rev"], 1), detail={"腰行程°/分钟": round(v["waist_travel"], 1)}),
        "ik": dict(score=S.comb(lin(v["ik_p50"], "ik_p50_ms"), lin(v["ik_p99"], "ik_p99_ms")),
                   value=None if v["ik_p50"] is None else round(v["ik_p50"], 1),
                   detail={"p99 ms": None if v["ik_p99"] is None else round(v["ik_p99"], 1)}),
    }
    out = {"name": name, "kind": "ik", "split": split, "asof": asof, "minutes": round(m, 1),
           "config": json.load(open(os.path.join(d, "config.json"))), "dims": dims}
    json.dump(out, open(os.path.join(d, f"score_{split}.json"), "w"), ensure_ascii=False, indent=1)
    return out


# ------------------------------------------------------------------------------------------------ walk
def run_walk(a):
    d = os.path.join(OUT, a.name)
    os.makedirs(d, exist_ok=True)
    params = json.loads(a.params or "{}")
    json.dump({"kind": "walk", "params": params, "dirs": a.dirs}, open(os.path.join(d, "config.json"), "w"))
    env = dict(os.environ, STEP_SIM_DELAY=os.environ.get("STEP_SIM_DELAY", "1"))
    procs = []
    for dr in a.dirs.split(","):
        c = ["nice", "-n", "19", sys.executable, os.path.join(HERE, "walk_sim.py"), "custom", "--params", json.dumps(params),
             "--dirs", dr, "--json", os.path.join(d, f"walk_{dr}.json")]
        while len([p for p in procs if p.poll() is None]) >= a.j:
            next(p for p in procs if p.poll() is None).wait()
        procs.append(subprocess.Popen(c, env=env, stdout=open(os.path.join(d, f"walk_{dr}.txt"), "w"),
                                      stderr=subprocess.STDOUT))
    if any(p.wait() != 0 for p in procs):
        sys.exit(f"walk sim failed: see {d}/walk_*.txt")
    print(json.dumps(score_walk(a.name), ensure_ascii=False, indent=1))


def score_walk(name):
    d = os.path.join(OUT, name)
    runs = []
    for f in sorted(os.listdir(d)):
        if f.startswith("walk_") and f.endswith(".json"):
            runs += json.load(open(os.path.join(d, f)))
    st = [s for r in runs for s in r["steps"]]
    if not st:
        return None
    med = {k: float(np.median([s[k] for s in st])) for k in ("roll", "roll_rate", "pitch", "pitch_rate", "w_roll", "w_yaw", "w_pitch", "dur")}
    extra = 100.0 * float(np.mean([s["extra"] > 0 for s in st]))
    falls = sum(r["fell"] for r in runs)
    per_dir = {}
    for r in runs:
        per_dir.setdefault(r["dir"], []).extend(r["steps"])
    out = {"name": name, "kind": "walk", "steps": len(st), "falls": falls,
           "config": json.load(open(os.path.join(d, "config.json"))),
           "dims": {"walking": dict(score=None if falls else S.comb(lin(med["roll"], "roll_pp_deg"), lin(med["roll_rate"], "roll_rate_dps"),
                                                                  lin(med["pitch"], "pitch_pp_deg"), lin(extra, "extra_pct")),
                                    value=round(med["roll"], 2),
                                    detail={"侧晃角速度°/s": round(med["roll_rate"], 1), "前后晃°": round(med["pitch"], 2),
                                            "前后晃角速度°/s": round(med["pitch_rate"], 1),
                                            "腰 roll/yaw/pitch°": [round(med["w_roll"], 1), round(med["w_yaw"], 1), round(med["w_pitch"], 1)],
                                            "多迈一步%": round(extra, 1), "摔倒": falls})},
           "per_dir": {k: {"n": len(v), "roll": round(float(np.median([s["roll"] for s in v])), 2),
                           "extra": sum(s["extra"] > 0 for s in v)} for k, v in per_dir.items()}}
    json.dump(out, open(os.path.join(d, "score.json"), "w"), ensure_ascii=False, indent=1)
    return out


# ------------------------------------------------------------------------------------------------ events
def events(a):
    """The worst events of a replayed IK candidate, with what each arm joint did, the hand targets and the waist."""
    d = os.path.join(OUT, a.name)
    cfg = json.load(open(os.path.join(d, "config.json")))
    rows = []
    for x in assets("train", a.asof or cfg.get("asof")):
        f = os.path.join(d, x["id"] + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        t, n, tgt, e, q = z["t"], [str(v) for v in z["names"]], z["tgt"], z["err"], z["q_rep"]
        arm = [i for i, v in enumerate(n) if "hand" not in v and not v.startswith("waist")]
        wst = [i for i, v in enumerate(n) if v.startswith("waist")]
        an = [n[i].replace("_joint", "") for i in arm]
        qa = q[:, arm]
        k, N = 0, len(t)
        while k < N:
            j = k
            while j + 1 < N and t[j + 1] - t[k] <= 0.5 and t[j + 1] - t[j] < 0.15:
                j += 1
            hit = None
            if j > k:
                dh = max(np.linalg.norm(tgt[j, :3] - tgt[k, :3]), np.linalg.norm(tgt[j, 7:10] - tgt[k, 7:10]))
                dq = qa[j] - qa[k]
                if a.kind == "swing" and dh < 0.05 and np.abs(dq).max() > math.radians(25):
                    hit = float(np.degrees(np.abs(dq).max()))
                elif a.kind == "outward" and dh < 0.06 and max(sg * dq[i] for i, sg in S.OUTWARD) > math.radians(15):
                    hit = float(np.degrees(max(sg * dq[i] for i, sg in S.OUTWARD)))
                elif a.kind == "err" and np.nanmax(e[k:j + 1]) > 5:
                    hit = float(np.nanmax(e[k:j + 1]))
                elif a.kind == "jerk" and j - k >= 2:
                    dt = np.diff(t[k:j + 1])
                    v = np.diff(qa[k:j + 1], axis=0) / np.maximum(dt, 1e-3)[:, None]
                    if len(v) >= 2 and np.abs(np.diff(v, axis=0)).max() > 2.0:
                        hit = float(np.abs(np.diff(v, axis=0)).max())
            if hit is not None:
                dq = np.degrees(qa[j] - qa[k])
                top = np.argsort(-np.abs(dq))[:4]
                rows.append(dict(asset=x["id"], t=round(float(t[k] - t[0]), 2), size=round(hit, 1), dur=round(float(t[j] - t[k]), 2),
                                 joints={an[i]: round(float(dq[i]), 1) for i in top},
                                 elbow_lr=[round(float(qa[k, an.index("left_elbow")]), 2), round(float(qa[j, an.index("left_elbow")]), 2),
                                           round(float(qa[k, an.index("right_elbow")]), 2), round(float(qa[j, an.index("right_elbow")]), 2)],
                                 waist_deg=[round(float(v), 1) for v in np.degrees(q[j, wst] - q[k, wst])],
                                 tgt_l=[round(float(v), 3) for v in tgt[k, :3]], tgt_r=[round(float(v), 3) for v in tgt[k, 7:10]],
                                 hand_move_cm=round(100 * float(max(np.linalg.norm(tgt[j, :3] - tgt[k, :3]), np.linalg.norm(tgt[j, 7:10] - tgt[k, 7:10]))), 1),
                                 err_max=round(float(np.nanmax(e[k:j + 1])), 2)))
                k = j
                continue
            k += 1
    rows.sort(key=lambda r: -r["size"])
    print(json.dumps({"name": a.name, "kind": a.kind, "count": len(rows),
                      "note": "t = seconds into the asset; joints = largest joint changes (deg) over the event; elbow_lr = "
                              "[left start, left end, right start, right end] rad (1.41 = straight); waist_deg = yaw/roll/pitch "
                              "change; tgt_l/tgt_r = wrist targets (m, robot frame) at the start; err in cm + deg/20",
                      "top": rows[:a.top]}, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------------------------------------ gate
def load(name, split):
    d = os.path.join(OUT, name)
    for f in (f"score_{split}.json", "score.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def gate(base, cand, targets):
    """(passed, reasons): every target dimension +GATE_GAIN, no dimension -GATE_DROP, hard limits."""
    why, ok = [], True
    for k, (b, c) in ((k, (base["dims"].get(k), cand["dims"].get(k))) for k in base["dims"]):
        if b is None or c is None or b.get("score") is None or c.get("score") is None:
            continue
        if k == "ik":           # ms per frame depends on the shared workstation load: hard limit below, no +-5 rule
            why.append(f"ik {b['score']} → {c['score']}（只检查 p99 ≤ 50 ms）")
            continue
        dlt = c["score"] - b["score"]
        if k in targets and dlt < GATE_GAIN:
            ok = False
            why.append(f"目标维度 {k} 只变化 {dlt:+d} 分（要求 ≥ +{GATE_GAIN}）")
        elif k not in targets and dlt < -GATE_DROP:
            ok = False
            why.append(f"{k} 变差 {dlt:+d} 分（允许 ≥ -{GATE_DROP}）")
        else:
            why.append(f"{k} {b['score']} → {c['score']} ({dlt:+d})")
    ik = cand["dims"].get("ik") or {}
    p99 = (ik.get("detail") or {}).get("p99 ms")
    if p99 is not None and p99 > 50:
        ok = False
        why.append(f"逆解 p99 {p99} ms 超过 50 ms 预算")
    if cand.get("falls"):
        ok = False
        why.append(f"仿真摔倒 {cand['falls']} 次")
    return ok, why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ik")
    p.add_argument("name")
    p.add_argument("--split", default="train", choices=["train", "test", "all"])
    p.add_argument("--asof", default=None)
    p.add_argument("--smooth", default="1")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--ik", default=None, help="candidate dwbc_ik_global.py (default: the repo's)")
    p.add_argument("-j", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("walk")
    p.add_argument("name")
    p.add_argument("--params", default="{}")
    p.add_argument("--dirs", default="back,left,right,fwd,turn_left")
    p.add_argument("-j", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("name")
    p.add_argument("--split", default="train")
    p.add_argument("--asof", default=None)
    p = sub.add_parser("events")
    p.add_argument("name")
    p.add_argument("--kind", default="swing", choices=["swing", "outward", "err", "jerk"])
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--asof", default=None)
    p = sub.add_parser("gate")
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--split", default="train")
    p.add_argument("--target", default="")
    a = ap.parse_args()
    if a.cmd in ("ik", "walk"):
        if not a.force and real_chain_running():
            sys.exit("真机链路正在运行：仿真暂停（--force 可强制）")
        (run_ik if a.cmd == "ik" else run_walk)(a)
    elif a.cmd == "events":
        events(a)
    elif a.cmd == "show":
        s = load(a.name, a.split)
        if s and s["kind"] == "ik":
            s = score_ik(a.name, a.split, a.asof or s.get("asof")) or s
        print(json.dumps(s, ensure_ascii=False, indent=1))
    else:
        b, c = load(a.base, a.split), load(a.cand, a.split)
        if b is None or c is None:
            sys.exit("缺评分：先跑 ik / walk")
        ok, why = gate(b, c, [x for x in a.target.split(",") if x])
        print(("PASS" if ok else "FAIL") + "\n  " + "\n  ".join(why))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
