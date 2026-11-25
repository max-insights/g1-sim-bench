"""Motor-gain overrides for the decoupled-WBC REAL chain, editable from the Mac app (hub dwbc_gains_get / _set,
2026-11-23). Read once by dwbc_run.py when the control loop starts: a change applies at the next start of
Decoupled WBC, never while the robot stands (gains are not hot-swapped on purpose).

waist_pitch_kd: upstream sets 5 - 10 = -5 on the real robot ("waist pitch is over-damped in sim; reduce KD by 10 on the
real robot to avoid sluggish response"). Negative damping amplifies motion: in MuJoCo -5 made deep squats slide
33-35 cm and stand-ups drift / fall with 20 ms delay, 0 and +5 did not; NVIDIA's own SONIC deploy uses about +1.8.
None / missing = upstream value.

Precedence in dwbc_run.py: DWBC_WAIST_PITCH_KD (environment, manual tests) > this file > upstream.
"""
import json
import os
import time

GAINS_FILE = (os.environ.get("DWBC_GAINS_FILE")            # tests point this elsewhere
              or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dwbc_gains.json"))
UPSTREAM = {"waist_pitch_kd": -5.0}
LIMITS = {"waist_pitch_kd": (-5.0, 10.0)}
NAMES_CN = {"waist_pitch_kd": "腰俯仰阻尼 kd"}


def validate(raw):
    """{"waist_pitch_kd": number | None} -> (values, [errors]); None means "use upstream"."""
    out, errs = {}, []
    if not isinstance(raw, dict):
        return {}, ["gains 必须是对象"]
    for k, v in raw.items():
        if k not in UPSTREAM:
            errs.append(f"未知参数 {k!r}")
            continue
        if v is None:
            out[k] = None
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errs.append(f"{NAMES_CN[k]}：需要数字或 null")
            continue
        lo, hi = LIMITS[k]
        if not (lo <= x <= hi):
            errs.append(f"{NAMES_CN[k]}：{x} 超出 {lo}–{hi}")
        else:
            out[k] = round(x, 2)
    return out, errs


def load(path=None):
    """-> (values {key: float | None}, errors). A bad file is ignored as a whole (= upstream)."""
    path = path or GAINS_FILE
    base = {k: None for k in UPSTREAM}
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return base, []
    except (OSError, ValueError) as e:
        return base, [f"{path} 读不了：{e}"]
    got, errs = validate({k: v for k, v in raw.items() if k in UPSTREAM})
    if errs:
        return base, errs
    return dict(base, **got), []


def load_meta(path=None):
    try:
        with open(path or GAINS_FILE) as f:
            raw = json.load(f)
        return {"updated": raw.get("updated"), "by": raw.get("by")}
    except Exception:
        return {"updated": None, "by": None}


def save(values, path=None, by=""):
    """values {key: float | None}; all None -> file removed (= upstream)."""
    path = path or GAINS_FILE
    if not values or all(v is None for v in values.values()):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    doc = {k: v for k, v in values.items() if k in UPSTREAM}
    doc["updated"] = time.time()
    doc["by"] = by
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
