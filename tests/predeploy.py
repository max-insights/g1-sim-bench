"""Pre-deploy simulation gate (formal process from 2026-11-27): every change to the teleop arm IK must pass this before
it runs on the real robot. Compares the candidate with the configuration that runs on the robot NOW (dwbc_env.sh
defaults, mapped to the IK exactly as dwbc_run.py does) on ALL recorded operator input (tests/ik_bench.py assets,
train + test, newest included), with walking under the candidate's upper body, through gate v2 (tests/sim_eval.py).

    python tests/predeploy.py NAME [--env DWBC_IK_X=V ...] [--set ATTR=V ...] [--ik FILE] [--target jumps,track]
    python tests/predeploy.py --live            # (re)score the live configuration only

  --env   override a dwbc_env.sh variable the way it would be changed for the robot (DWBC_IK_ELBOW_MAX=1.0 ...)
  --set   extra GlobalBodyIKSolver class attributes (a new knob not wired to an env variable yet)
  --ik    candidate solver file (default: this repo's dwbc_ik_global.py)
  --target dimensions the change is meant to improve (each must gain >= 5)

PASS (exit 0) is required before the robot test; the report goes to $IK_BENCH/results/predeploy/<date>_<NAME>.{json,txt}
and is what the person approving the robot test reads. FAIL (exit 1) = do not take it to the robot; the reasons say
which rule broke. Not covered here: leg-policy / gains / network changes (use `sim_eval.py walk` and the real-robot
scorecard); the IK time check is load-sensitive, run it when the workstation is quiet.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ik_bench as B          # noqa: E402
import sim_eval as E          # noqa: E402

OUT = os.path.join(B.ROOT, "results", "predeploy")
IK_VARS = ("DWBC_IK_SMOOTH", "DWBC_IK_LOCK_ROLL", "DWBC_IK_ROLL_MAX_DEG", "DWBC_IK_ELBOW_MAX", "DWBC_IK_WAIST_SOFT",
           "DWBC_IK_POSTURE_COST", "DWBC_IK_GLOBAL")


def live_env():
    """the IK variables a robot start gets: dwbc_env.sh defaults (the caller's own overrides ignored)."""
    clean = {k: v for k, v in os.environ.items() if k not in IK_VARS}
    r = subprocess.run(["bash", "-c", f"source {REPO}/dwbc_env.sh >/dev/null 2>&1; env"], capture_output=True, text=True,
                       env=clean)
    env = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    return {k: env.get(k) for k in IK_VARS}


def env_to_args(env):
    """dwbc_run.py's mapping of the env variables to GlobalBodyIKSolver (keep in sync with dwbc_run.py)."""
    sets = [f"LOCK_WAIST_ROLL={env.get('DWBC_IK_LOCK_ROLL') == '1'}",
            f"WAIST_ROLL_MAX_DEG={float(env.get('DWBC_IK_ROLL_MAX_DEG') or 0)}",
            f"ELBOW_MAX={float(env.get('DWBC_IK_ELBOW_MAX') or 0)}"]
    if env.get("DWBC_IK_WAIST_SOFT") == "1":
        sets += ["WAIST_PRIORITY=False", "WAIST_POSTURE_W=12.0", "WAIST_NULL_MULT=3.0"]
    if env.get("DWBC_IK_GLOBAL") != "1":
        sys.exit("DWBC_IK_GLOBAL is not 1: the replay only covers the global solver")
    return sets, ("1" if env.get("DWBC_IK_SMOOTH") == "1" else "0"), env.get("DWBC_IK_POSTURE_COST") or "0.01"


def fingerprint(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8]


def evaluate(name, sets, smooth, posture, ik):
    if E.load(name, "all") and "walking" in (E.load(name, "all") or {}).get("dims", {}):
        return E.score_ik(name, "all", None)
    c = [sys.executable, os.path.join(HERE, "sim_eval.py"), "ik", name, "--split", "all", "--smooth", smooth, "--walk", "-j", "6",
         "--ik", ik] + [x for s in sets for x in ("--set", s)]
    r = subprocess.run(c, capture_output=True, text=True, env=dict(os.environ, DWBC_IK_POSTURE_COST=posture))
    if r.returncode:
        sys.exit(f"replay failed ({name}):\n{(r.stdout + r.stderr)[-2000:]}")
    return E.load(name, "all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--ik", default=os.path.join(REPO, "dwbc_ik_global.py"))
    ap.add_argument("--target", default="")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if E.real_chain_running():
        sys.exit("真机链路正在运行：仿真暂停")
    ik = os.path.realpath(a.ik)
    ik_sha = hashlib.sha1(open(ik, "rb").read()).hexdigest()[:8]
    assets = [x["id"] for x in B.chosen("all")]
    lenv = live_env()
    lsets, lsmooth, lpost = env_to_args(lenv)
    live_ik = os.path.join(REPO, "dwbc_ik_global.py")
    live_name = "live_" + fingerprint(lsets, lsmooth, lpost, hashlib.sha1(open(live_ik, "rb").read()).hexdigest(), assets)
    print(f"现在真机的配置：{lenv}\n  → {live_name}", flush=True)
    base = evaluate(live_name, lsets, lsmooth, lpost, live_ik)
    if a.live or not a.name:
        print(json.dumps(base, ensure_ascii=False, indent=1))
        return
    cenv = dict(lenv)
    for kv in a.env:
        k, v = kv.split("=", 1)
        if k not in IK_VARS:
            sys.exit(f"--env {k}: not an IK variable ({', '.join(IK_VARS)})")
        cenv[k] = v
    csets, csmooth, cpost = env_to_args(cenv)
    csets += a.set
    cname = "pre_" + a.name + "_" + fingerprint(csets, csmooth, cpost, ik_sha, assets)
    print(f"候选：env 改动 {a.env or '无'}，额外参数 {a.set or '无'}，逆解文件 {ik}（{ik_sha}）\n  → {cname}", flush=True)
    cand = evaluate(cname, csets, csmooth, cpost, ik)
    ok, why = E.gate(base, cand, [x for x in a.target.split(",") if x])
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, time.strftime("%Y%m%d_%H%M") + "_" + a.name)
    rows = [f"{'维度':8s} {'现在真机':>8s} {'候选':>6s}"] + [
        f"{k:8s} {str((base['dims'].get(k) or {}).get('score')):>8s} {str((cand['dims'].get(k) or {}).get('score')):>6s}"
        for k in ("jumps", "jitter", "track", "waist", "ik", "walking")]
    txt = "\n".join([f"上线前仿真门禁：{a.name}  {'PASS' if ok else 'FAIL'}",
                     f"时间 {time.strftime('%Y-%m-%d %H:%M')}，数据 {len(assets)} 段录像（全部，{base['minutes']} 分钟）",
                     f"现在真机 {live_name}：{lenv}", f"候选 {cname}：env {a.env}，set {a.set}，ik {ik}（{ik_sha}）", ""]
                    + rows + [""] + why)
    open(stem + ".txt", "w").write(txt + "\n")
    json.dump({"ok": ok, "why": why, "live": base, "cand": cand, "env": a.env, "set": a.set, "ik": ik, "ik_sha": ik_sha},
              open(stem + ".json", "w"), ensure_ascii=False, indent=1)
    print(txt + f"\n\n报告：{stem}.txt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
