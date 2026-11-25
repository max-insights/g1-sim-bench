"""Offline test of dwbc_step_ext (no DDS / ROS): control-side patch through keys on the real policy class in MuJoCo,
and the PICO flick / click logic on a stand-in streamer.   python tests/step_ext_test.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ["DWBC_QUIET"] = "1"
import tempfile  # noqa: E402
TMP = tempfile.mkdtemp()
os.environ["DWBC_STEP_PROFILE_FILE"] = os.path.join(TMP, "step_profiles.json")   # never the real settings file
import dwbc_step_ext  # noqa: E402
import dwbc_step  # noqa: E402

dwbc_step_ext.install_control()
import step_sim  # noqa: E402

ok = True
sim = step_sim.Sim()


def yaw():
    w, x, y, z = sim.d.qpos[3:7]
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


step_sim.stand(sim, 2.5)
assert sim.pol._dwbc_step is not None
for key, want in (("t", "x+"), ("g", "x-"), ("f", "y+"), ("h", "y-"), ("r", "w+"), ("y", "w-")):
    lc = step_sim.LiftCounter(sim.feet_world_z())
    b0 = sim.d.qpos[0:2].copy()
    yaw0 = yaw()
    sim.pol.handle_keyboard_button(key)
    for _ in range(int(2.5 / 0.02)):
        sim.nav[:] = 0.0              # no stick: only the step override may move the legs
        sim.tick()
        lc.update(sim.t, sim.feet_world_z())
    d = 100 * (sim.d.qpos[0:2] - b0)
    dyaw = (yaw() - yaw0 + 180) % 360 - 180
    v = {"x": d[0], "y": d[1], "w": dyaw / 8}[want[0]]          # turn: at least 8 deg (target 15, by angle)
    feet = lc.lifts == [1, 1] if want[0] != "w" else 1 <= sum(lc.lifts) <= 3
    good = feet and (v > 1 if want[1] == "+" else v < -1) and sim.d.qpos[2] > 0.6
    ok &= good
    print(f"key {key}: lifts L/R {lc.lifts}  moved x {d[0]:+.1f} y {d[1]:+.1f} cm yaw {dyaw:+.1f} deg  {'OK' if good else 'FAIL'}")
# settings file (Mac app): hot reload at the next request, bad file ignored as a whole
dwbc_step.save_profiles({"fwd": (0.30, "touchdown", 2)}, by="test")
ss = sim.pol._dwbc_step
sim.pol.handle_keyboard_button("t")
good = ss._cmd[0] == 0.30 and ss._stop_on == "touchdown"
ok &= good
print("settings file hot-reloaded (fwd 0.30, touchdown)", "OK" if good else f"FAIL {ss._cmd} {ss._stop_on}")
for _ in range(int(3.0 / 0.02)):
    sim.nav[:] = 0.0
    sim.tick()
import json, time as _t  # noqa: E401,E402
_t.sleep(0.01)
json.dump({"fwd": {"speed": 9.0, "stop_on": "lift", "count": 2}}, open(dwbc_step.PROFILE_FILE, "w"))
sim.pol.handle_keyboard_button("t")
good = abs(ss._cmd[0] - dwbc_step.PROFILES["fwd"][0]) < 1e-9
ok &= good
print("out-of-range file ignored -> defaults", "OK" if good else f"FAIL {ss._cmd}")
for _ in range(int(3.0 / 0.02)):
    sim.nav[:] = 0.0
    sim.tick()
p, src, errs = dwbc_step.load_profiles()
print("  load_profiles ->", src, errs)
dwbc_step.save_profiles(None)
ok &= not os.path.exists(dwbc_step.PROFILE_FILE)

# hold turning on the real policy class: hold left 2 s (as the teleop sends it every frame), then release
ss = sim.pol._dwbc_step
yaw0 = yaw()
for k in range(int(4.0 / 0.02)):
    if k < 100:
        ss.set_hold(sim.t, 1.0)
    elif k == 100:
        ss.set_hold(sim.t, 0.0)
        yaw_rel = yaw()
    sim.nav[:] = 0.0
    sim.tick()
dy = (yaw() - yaw0 + 180) % 360 - 180
after = (yaw() - yaw_rel + 180) % 360 - 180
good = dy > 25 and abs(after) < 10 and not ss.busy and sim.d.qpos[2] > 0.6
ok &= good
print(f"hold left 2 s then release: turned {dy:+.1f} deg, {after:+.1f} after the release", "OK" if good else "FAIL")
# stale hold (teleop link lost while held) -> stops by itself
for k in range(int(3.0 / 0.02)):
    if k == 0:
        ss.set_hold(sim.t, -1.0)
    sim.nav[:] = 0.0
    sim.tick()
good = not ss.busy and any(e == "release" and "no stick update" in d for _, e, d in ss.log)
ok &= good
print("hold without updates (link lost) -> released and stopped", "OK" if good else "FAIL")

# leg policy off -> request ignored
sim.pol.use_policy_action = False
sim.pol.handle_keyboard_button("t")
ok &= not sim.pol._dwbc_step.busy
print("policy off: request ignored", "OK" if not sim.pol._dwbc_step.busy else "FAIL")

# ---- teleop side: flick / click logic on a stand-in for PicoStreamer
from decoupled_wbc.control.teleop.streamers import pico_streamer as PS  # noqa: E402


class Out:
    def __init__(self, h):
        self.control_data = {"navigate_cmd": [0.3, 0.0, 0.0], "base_height_command": h}


def fake_orig(self, pd):              # the original needs a live PICO: upstream height logic = +-0.01 per frame
    if pd.get("Y"):
        self.current_base_height = min(0.74, self.current_base_height + 0.01)
    if pd.get("X"):
        self.current_base_height = max(0.2, self.current_base_height - 0.01)
    return Out(self.current_base_height)


PS.PicoStreamer._generate_unified_raw_data = fake_orig
clock = [1000.0]
dwbc_step_ext._mono = lambda: clock[0]
dwbc_step_ext.install_teleop()
fake = PS.PicoStreamer.__new__(PS.PicoStreamer)
fake.current_base_height = 0.74
gen = PS.PicoStreamer._generate_unified_raw_data


def frame(x=0.0, y=0.0, click=False, rx=0.0, X=False, Y=False, dt=0.05):
    clock[0] += dt
    return gen(fake, {"left_axis_click": click, "left_joystick": [x, y], "right_joystick": [rx, 0.0],
                      "X": X, "Y": Y}).control_data


cd = frame(y=0.9)
ok &= cd["dwbc_step"]["seq"] == 0 and cd["navigate_cmd"][0] == 0.3          # mode off: stick passes through
frame(click=True); frame()                                                   # click -> mode on
seqs = []
for x, y in [(0, 0.9), (0, 0.95), (0, 0.1), (0, -0.8), (0, 0), (-0.7, 0.1), (0, 0), (0.3, 0.4), (0.9, 0), (0, 0)]:
    cd = frame(x, y)
    seqs.append((cd["dwbc_step"]["seq"], cd["dwbc_step"]["dir"], cd["navigate_cmd"]))
print("teleop frames:", [(s, d) for s, d, _ in seqs])
want = [(1, "fwd"), (1, "fwd"), (1, "fwd"), (2, "back"), (2, "back"), (3, "left"), (3, "left"), (3, "left"),
        (4, "right"), (4, "right")]
good = [(s, d) for s, d, _ in seqs] == want and all(n == [0.0, 0.0, 0.0] for _, _, n in seqs)
ok &= good
print("teleop flicks: one seq per flick, stick walking off in step mode", "OK" if good else "FAIL")
frame(click=True); cd = frame(y=0.9)
good = cd["dwbc_step"]["mode"] is False and cd["navigate_cmd"][0] == 0.3
ok &= good
print("click again -> mode off, stick walks again", "OK" if good else "FAIL")
frame(click=True); frame()                                                   # mode on again
s0 = frame()["dwbc_step"]["seq"]
frame(rx=-0.9); frame(rx=-0.95); r1 = frame()["dwbc_step"]                 # 0.1 s push -> flick on release
frame(rx=0.8); r2 = frame()["dwbc_step"]
good = (r1["dir"], r2["dir"]) == ("turn_left", "turn_right") and r1["seq"] == s0 + 1 and r2["seq"] == s0 + 2 \
    and r1["turn_hold"] == 0 and r2["turn_hold"] == 0
ok &= good
print("right stick flicks -> turn_left / turn_right, one per flick (sent on release)", "OK" if good else f"FAIL {r1} {r2}")
hv = [frame(rx=-1.0)["dwbc_step"] for _ in range(14)]                       # held 0.7 s
rel = frame()["dwbc_step"]
good = (hv[5]["turn_hold"] == 0 and hv[9]["turn_hold"] == 1.0 and hv[-1]["seq"] == r2["seq"]
        and rel["turn_hold"] == 0 and rel["seq"] == r2["seq"])
ok &= good
print("right stick held 0.7 s -> turn_hold 1.0 after 0.4 s, no flick; release -> 0",
      "OK" if good else f"FAIL {[h['turn_hold'] for h in hv]} {rel}")
half = [frame(rx=0.7)["dwbc_step"]["turn_hold"] for _ in range(12)]; frame()
good = abs(half[-1] + (0.7 - 0.25) / 0.75) < 1e-9
ok &= good
print(f"right stick held 70 % right -> turn_hold {half[-1]:+.2f} (proportional, negative = right)", "OK" if good else "FAIL")

# rise speed (settings file): Y from 0.5 m, 20 Hz
def rise(n=200):
    fake.current_base_height = 0.5
    hs = []
    for _ in range(n):
        hs.append(frame(Y=True)["base_height_command"])
    return hs
hs = rise()
t_full = next(i for i, h in enumerate(hs) if h >= 0.74 - 1e-9) * 0.05
good = abs(t_full - 1.2) < 0.11                                           # default 0.2 m/s: 0.24 m in ~1.2 s
ok &= good
print(f"rise default: 0.50 -> 0.74 m in {t_full:.2f} s", "OK" if good else "FAIL")
dwbc_step.save_profiles({}, by="test", extras={"rise": {"speed": 0.05, "slow_zone": 0.05, "slow_speed": 0.02}})
hs = rise(400)
t_full = next(i for i, h in enumerate(hs) if h >= 0.74 - 1e-9) * 0.05
want = 0.19 / 0.05 + 0.05 / 0.02
good = abs(t_full - want) < 0.2 and max(abs(b - a) for a, b in zip(hs, hs[1:])) <= 0.05 * 0.05 + 1e-9
ok &= good
print(f"rise 0.05 m/s + last 5 cm at 0.02: 0.50 -> 0.74 m in {t_full:.2f} s (want {want:.2f})", "OK" if good else "FAIL")
fake.current_base_height = 0.74
d0 = [frame(X=True)["base_height_command"] for _ in range(5)]
good = abs(d0[-1] - 0.69) < 1e-9                                           # going down is not limited
ok &= good
print("going down unchanged (0.01 per frame)", "OK" if good else f"FAIL {d0}")
dwbc_step.save_profiles({}, by="test", extras={"turn": {"step_deg": 30, "hold_s": 0.2}})
x, src, errs = dwbc_step.load_extras()
good = x["turn"]["step_deg"] == 30 and x["rise"]["speed"] == 0.2 and src == "file"
_ = frame()
hv = [frame(rx=-1.0)["dwbc_step"]["turn_hold"] for _ in range(6)]; frame()
good &= hv[1] == 0 and hv[5] == 1.0                                       # hold_s 0.2 hot-reloaded (0.4: none)
ok &= good
print("turn settings hot-reloaded (hold_s 0.2)", "OK" if good else f"FAIL {x} {hv}")
json.dump({"turn": {"step_deg": 90}}, open(dwbc_step.PROFILE_FILE, "w"))
x, src, errs = dwbc_step.load_extras()
good = x["turn"]["step_deg"] == 15 and bool(errs)
ok &= good
print("bad turn setting (90 deg) -> defaults", errs, "OK" if good else "FAIL")
dwbc_step.save_profiles(None)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
