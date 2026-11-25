"""Run a decoupled-WBC entry point with our overrides (the upstream files stay untouched).

    python dwbc_run.py control [tyro args ...]     # decoupled_wbc ... run_g1_control_loop
    python dwbc_run.py teleop  [tyro args ...]     # decoupled_wbc ... run_teleop_policy_loop

DWBC_DDS_DOMAIN  Unitree DDS domain for the control loop and its MuJoCo sim (the yaml hard-codes 0). The sim uses 7:
                 another session's fake_g1.py sits on lo / domain 0 and fake_g1_phys.py on domain 9.
DWBC_ENABLE_WAIST "1": add --enable-waist (waist in the arm IK -> torso lean to the leg policy).
DWBC_IK_POSTURE_COST e.g. "0.1": arm IK posture task weight for the teleop loop (upstream 0.01).
DWBC_IK_GLOBAL   "1": multi-start (global) arm IK for the teleop loop (dwbc_ik_global.py).
DWBC_ABXY_RESET  "1": A+B+X+Y returns slowly to the standing pose (dwbc_teleop_ext.py).
DWBC_KP_SCALE    optional, e.g. "0.8": scales every MOTOR_KP (not used by default).
DWBC_WAIST_PITCH_KD e.g. "0": waist pitch kd override (real upstream: -5), unset = upstream.
DWBC_STEP        "1": single-step mode (dwbc_step_ext.py), off by default.
DWBC_LATENCY     "0" turns off the per-hop latency recording (dwbc_latency.py; on by default).
"""
import os
import runpy
import sys

import decoupled_wbc.control.main.teleop.configs.configs as C

_orig = C.override_wbc_config


def _override(wbc_config, config, missed_keys_only=False):
    out = _orig(wbc_config, config, missed_keys_only)
    dom = os.environ.get("DWBC_DDS_DOMAIN")
    if dom is not None:
        out["DOMAIN_ID"] = int(dom)
    s = os.environ.get("DWBC_KP_SCALE")
    if s:
        out["MOTOR_KP"] = [float(k) * float(s) for k in out["MOTOR_KP"]]
    # DWBC_WAIST_PITCH_KD (2026-11-23, unset = upstream): on the real robot upstream sets the waist pitch kd to
    # 5 - 10 = -5 ("sim2real gap", configs.py). In MuJoCo -5 makes deep squats slide 33-35 cm and stand-ups drift /
    # fall with 20 ms delay; 0 and +5 do not (tests/step_sim.py squat). Candidate fix for the step after standing up.
    # Mac app (设置 → 腰部阻尼, hub dwbc_gains_set): dwbc_gains.json, REAL robot only, read once here at start.
    # Precedence: DWBC_WAIST_PITCH_KD (environment) > dwbc_gains.json > upstream.
    wk, src = os.environ.get("DWBC_WAIST_PITCH_KD"), "env DWBC_WAIST_PITCH_KD"
    if wk in (None, "") and out.get("ENV_TYPE") == "real":
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import dwbc_gains

            vals, errs = dwbc_gains.load()
            for e in errs:
                print(f"[dwbc_run] dwbc_gains.json ignored: {e}", flush=True)
            wk, src = vals.get("waist_pitch_kd"), "dwbc_gains.json (Mac app)"
        except Exception as e:  # noqa: BLE001 - never block the start over the override file
            print(f"[dwbc_run] dwbc_gains.json not read: {e!r}", flush=True)
            wk = None
    if wk not in (None, ""):
        out["MOTOR_KD"] = list(out["MOTOR_KD"])
        out["MOTOR_KD"][14] = float(wk)
    else:
        src = "upstream"
    print(f"[dwbc_run] waist pitch kd={out['MOTOR_KD'][14]} ({src})", flush=True)
    print(f"[dwbc_run] DOMAIN_ID={out.get('DOMAIN_ID')} INTERFACE={out.get('INTERFACE')} ENV_TYPE={out.get('ENV_TYPE')}",
          flush=True)
    return out


C.override_wbc_config = _override

if os.environ.get("DWBC_DRY") == "1":
    # dry run against dwbc_fake_robot.py: no Unitree MotionSwitcher service there. Upstream's real-mode
    # BodyStateProcessor calls CheckMode/ReleaseMode at start (it releases the mode itself on the robot).
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    MotionSwitcherClient.CheckMode = lambda self: (0, {"name": ""})
    MotionSwitcherClient.ReleaseMode = lambda self: (0, None)
    print("[dwbc_run] DRY: MotionSwitcher stubbed (no mode active)", flush=True)

MODULES = {
    "control": "decoupled_wbc.control.main.teleop.run_g1_control_loop",
    "teleop": "decoupled_wbc.control.main.teleop.run_teleop_policy_loop",
}
if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
    sys.exit(__doc__)
mod = MODULES[sys.argv[1]]
# DWBC_IK_POSTURE_COST (default 0.1 in dwbc_env.sh since 2026-11-17; upstream 0.01): weight of the arm IK posture task.
# At 0.01 the differential IK gets stuck: an elbow pinned at its limit (+120 deg, wrist 5 cm off) or, with the waist
# in the IK, a 16-30 deg BACKWARD lean that never recovers after a fast / stalled target jump. 0.1 removed every case
# in the offline study and still leans to the 30 deg waist limit (scratchpad ik_jump2.py / ik_lean_pc.py).
# DWBC_IK_GLOBAL=1: multi-start arm IK (dwbc_ik_global.py next to this file) replaces upstream BodyIKSolver in the
# teleop loop: extra candidate solutions re-seeded from fixed poses, best one wins, switches blend smoothly.
if os.environ.get("DWBC_IK_GLOBAL") == "1" and sys.argv[1] == "teleop":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import decoupled_wbc.control.teleop.teleop_retargeting_ik as _TRI
    from dwbc_ik_global import GlobalBodyIKSolver

    GlobalBodyIKSolver.verbose = True
    _TRI.BodyIKSolver = GlobalBodyIKSolver
    print("[dwbc_run] arm IK: multi-start global solver (dwbc_ik_global.py)", flush=True)
# DWBC_ABXY_RESET=1 (2026-11-18): A+B+X+Y after the first press slowly returns arms + waist to the standing pose
# (dwbc_teleop_ext.py; the streamer part is patch_reset.py in pico_streamer.py)
if os.environ.get("DWBC_ABXY_RESET") == "1" and sys.argv[1] == "teleop":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dwbc_teleop_ext

    dwbc_teleop_ext.install()
    print("[dwbc_run] A+B+X+Y: leg policy on + slow return to standing (dwbc_teleop_ext.py)", flush=True)
# DWBC_STEP=1 (2026-11-23, off by default): single-step mode (dwbc_step_ext.py). PICO left stick click toggles it, a
# flick = one step (each foot lifts once); keys t/g/f/h on the control loop. Control loop and teleop must agree.
if os.environ.get("DWBC_STEP") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dwbc_step_ext

    if sys.argv[1] == "control":
        dwbc_step_ext.install_control()
    else:
        dwbc_step_ext.install_teleop()
_pc = os.environ.get("DWBC_IK_POSTURE_COST")
if _pc and sys.argv[1] == "teleop":
    from decoupled_wbc.control.teleop.solver.body import body_ik_solver_settings as _BS

    _bs_init = _BS.BodyIKSolverSettings.__init__

    def _bs_init_tuned(self):
        _bs_init(self)
        self.posture_cost = float(_pc)

    _BS.BodyIKSolverSettings.__init__ = _bs_init_tuned
    print(f"[dwbc_run] arm IK posture_cost={float(_pc)} (upstream 0.01)", flush=True)
# DWBC_LATENCY (default on, 2026-11-20): per-hop latency recording of the PICO -> robot chain (dwbc_latency.py next to
# this file; lat_*.csv in the run's log dir; live view: python3 dwbc_latency.py watch). Installed last so it wraps the
# patches above. It only times calls; the one thing it adds is a "lat" entry (stamps) in the teleop goal message,
# which the control loop does not read. "0" turns it off.
if os.environ.get("DWBC_LATENCY", "1") != "0":
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dwbc_latency

        dwbc_latency.install(sys.argv[1])
    except Exception as _e:  # noqa: BLE001 - never block the control chain over the monitor
        print(f"[dwbc_run] latency monitor not installed: {_e!r}", flush=True)
sys.argv = [mod] + sys.argv[2:]
# DWBC_ENABLE_WAIST=1 (default in dwbc_env.sh since 2026-11-17): waist joints join the arm IK and the resulting torso
# lean is passed to the leg policy as its rpy command (upstream --enable-waist). Control loop and teleop must agree.
if os.environ.get("DWBC_ENABLE_WAIST") == "1" and not any("enable-waist" in a for a in sys.argv):
    sys.argv.append("--enable-waist")
print(f"[dwbc_run] {sys.argv[0].rsplit('.', 1)[-1]} enable_waist={'--enable-waist' in sys.argv}", flush=True)
runpy.run_module(mod, run_name="__main__")
