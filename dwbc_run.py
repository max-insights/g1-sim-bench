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
_pc = os.environ.get("DWBC_IK_POSTURE_COST")
if _pc and sys.argv[1] == "teleop":
    from decoupled_wbc.control.teleop.solver.body import body_ik_solver_settings as _BS

    _bs_init = _BS.BodyIKSolverSettings.__init__

    def _bs_init_tuned(self):
        _bs_init(self)
        self.posture_cost = float(_pc)

    _BS.BodyIKSolverSettings.__init__ = _bs_init_tuned
    print(f"[dwbc_run] arm IK posture_cost={float(_pc)} (upstream 0.01)", flush=True)
sys.argv = [mod] + sys.argv[2:]
# DWBC_ENABLE_WAIST=1 (default in dwbc_env.sh since 2026-11-17): waist joints join the arm IK and the resulting torso
# lean is passed to the leg policy as its rpy command (upstream --enable-waist). Control loop and teleop must agree.
if os.environ.get("DWBC_ENABLE_WAIST") == "1" and not any("enable-waist" in a for a in sys.argv):
    sys.argv.append("--enable-waist")
print(f"[dwbc_run] {sys.argv[0].rsplit('.', 1)[-1]} enable_waist={'--enable-waist' in sys.argv}", flush=True)
runpy.run_module(mod, run_name="__main__")
