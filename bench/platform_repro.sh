#!/bin/bash
# Reproducibility check of the simulation bench on MaxMerlin (2026-11-27): replays the starting configuration and the
# human fix (elbow 0.9 + soft waist) on train / test / hidden1126 with walking, exactly as the workstation scorer
# (tag bench-ik1125-v1.2-score), and compares every dimension except IK time with the workstation scores.
set -euo pipefail
cd "$(dirname "$0")/.."
J=${J:-12}
B="--set LOCK_WAIST_ROLL=1 --set WAIST_ROLL_MAX_DEG=10"
echo "nproc $(nproc)  python $(python3 --version)  commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
for sp in train test hidden1126; do
  python3 tests/sim_eval.py ik pf_base  --split $sp --asof 20261126 $B --ik bench/ik1125/base_solver.py --walk -j $J > /dev/null
  python3 tests/sim_eval.py ik pf_human --split $sp --asof 20261126 $B --set ELBOW_MAX=0.9 --set WAIST_PRIORITY=0 \
      --set WAIST_POSTURE_W=12.0 --set WAIST_NULL_MULT=3.0 --walk -j $J > /dev/null
  echo "done $sp $(date +%T)"
done
python3 - <<'PY'
import json, subprocess, sys
exp = json.load(open("bench/ik1125/expected_workstation.json"))
bad = 0
for ref, name in (("base", "pf_base"), ("human", "pf_human")):
    for sp, want in exp[ref].items():
        out = subprocess.run([sys.executable, "tests/sim_eval.py", "show", name, "--split", sp], capture_output=True, text=True).stdout
        got = json.loads(out[out.index("{"):])["dims"]
        g = {k: got[k]["score"] for k in want}
        ik = got["ik"]["detail"]
        ok = g == want
        bad += not ok
        print(f"{ref:6s} {sp:11s} {'MATCH' if ok else 'DIFF '}  platform {g}  workstation {want}  ik cpu p99 {ik.get('p99 ms')} ms")
print("REPRO", "PASS" if not bad else f"FAIL ({bad} rows differ)")
sys.exit(1 if bad else 0)
PY
