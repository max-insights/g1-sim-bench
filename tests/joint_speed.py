#!/usr/bin/env python3
"""Joint speeds in lat_track.csv (50 Hz, 14 arm joints): how often the command sits at the speed cap, measured speeds."""
import csv
import sys

for d in sys.argv[1:]:
    rows = list(csv.DictReader(open(d + "/lat_track.csv")))
    t = [float(r["t"]) for r in rows]
    cmd = [[float(r[f"cmd{i}"]) for i in range(14)] for r in rows]
    meas = [[float(r[f"meas{i}"]) for i in range(14)] for r in rows]
    cap, n, vc_all, vm_all = 0, 0, [], []
    capped_runs, cur = [], 0
    for k in range(1, len(rows)):
        dt = t[k] - t[k - 1]
        if not 0.01 < dt < 0.04:
            continue
        vc = max(abs(cmd[k][i] - cmd[k - 1][i]) / dt for i in range(14))
        vm = max(abs(meas[k][i] - meas[k - 1][i]) / dt for i in range(14))
        n += 1
        vc_all.append(vc)
        vm_all.append(vm)
        if vc >= 2.85:
            cap += 1
            cur += 1
        elif cur:
            capped_runs.append(cur)
            cur = 0
    vc_all.sort()
    vm_all.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))]
    capped_runs.sort(reverse=True)
    print(d.split("/")[-1], "frames", n,
          f"cmd at cap {100 * cap / n:.2f}%",
          f"cmd speed p50/p99/max {q(vc_all, .5):.2f}/{q(vc_all, .99):.2f}/{vc_all[-1]:.2f}",
          f"meas speed p99/p999/max {q(vm_all, .99):.2f}/{q(vm_all, .999):.2f}/{vm_all[-1]:.2f}",
          "longest capped stretches (frames@50Hz)", capped_runs[:5])
