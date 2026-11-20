#!/usr/bin/env bash
# NVIDIA Decoupled WBC on the REAL G1 (workstation; robot on enp13s0; PICO via XRoboToolkit; Dex1 grippers).
#
#   ~/piper_ws/dwbc/dwbc_real.sh check                  read-only preflight (no command to the robot)
#   ~/piper_ws/dwbc/dwbc_real.sh start --confirmed [--no-teleop] [--no-grippers]
#   ~/piper_ws/dwbc/dwbc_real.sh key K ...              ']' leg policy ON (feet on the ground first!), 'o' leg policy
#                                                       OFF (legs then HOLD their angles: not a stop), 'l' arms follow on/off
#   ~/piper_ws/dwbc/dwbc_real.sh estop                  kill the control loop + DAMPING (kp 0 / kd 8), then stop the rest
#   ~/piper_ws/dwbc/dwbc_real.sh status
#
# `start` MOVES THE ROBOT: it releases Unitree's motion mode (the robot goes limp), ramps all 29 joints to the WBC
# default pose in 3 s, then the control loop holds the legs and ramps the arms to their start pose in 2 s.
# Only with the robot on the gantry (feet on the ground, rope slightly slack), the area clear and someone at the
# physical e-stop. --confirmed means a person confirmed exactly that; without it the script asks for "YES".
# Stops: physical e-stop, Unitree remote, the Mac app's 急停 (g1_hub 0.4: estop_direct.sh runs `dwbc_real.sh estop`
# first), `dwbc_real.sh estop`. NOT the PICO. The guard (dwbc_guard.py) damps when the control loop dies, on estop,
# and on motor temperature (casing 85 / winding 115 C for 2 s).
# Usually started from the Mac app: 待命 -> 控制器「Decoupled WBC」 (g1_hub runs `start --confirmed`).
# Dry run without the robot: DWBC_DRY=1 DWBC_IFACE=lo inside `unshare -rn` with 192.168.123.99 on lo and
# dwbc_fake_robot.py (see README.md).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/dwbc_env.sh"     # before set -u: conda's ROS activate scripts read unset variables
set -u
RUN=$HOME/piper_ws/dwbc/run
mkdir -p "$RUN" "$DWBC_LOGS"
IFACE=${DWBC_IFACE:-enp13s0}
DRY=${DWBC_DRY:-0}
[ "$DRY" = 1 ] && DWBC_LOGS=$DWBC_LOGS/dry   # dry runs never become the real chain's "latest" (the hub / app read it)
FLAG=/tmp/dwbc_estop
G1PY=$HOME/g1_venv/bin/python
SDK=$DWBC_REPO/external_dependencies/unitree_sdk2_python
LL="python -u $HERE/dwbc_lowlevel.py --iface $IFACE"

alive_pid() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
grp_kill() {
    local f=$1 p; [ -f "$f" ] || return 0; p=$(cat "$f")
    for sig in INT TERM KILL; do
        kill -0 "$p" 2>/dev/null || break
        kill -s $sig -- -"$p" 2>/dev/null
        for i in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
    done
    rm -f "$f"
}
start_bg() {
    local name=$1 log=$2; shift 2
    rm -f "$RUN/$name.pid"
    setsid bash -c 'echo $$ > "'"$RUN/$name.pid"'"; exec "$@"' _ "$@" > "$log" 2>&1 < /dev/null &
    for i in $(seq 40); do [ -f "$RUN/$name.pid" ] && break; sleep 0.05; done
}
QUIET=${DWBC_QUIET:-0}
say() { [ "$QUIET" = 1 ] || spd-say -l zh "$1" >/dev/null 2>&1 || true; }
sonic_running() {   # the SONIC chain (real or sim) uses the same robot / ZMQ / PICO
    pgrep -f "^./target/release/g1_deploy_onnx_ref" >/dev/null || pgrep -f "pico_manager_thread_server.py" >/dev/null \
        || pgrep -f "real_teleop.sh (arm|start)" >/dev/null
}

check() {
    local L=${1:-/tmp}
    $LL check --json "$L/check.json"
}

start() {
    local confirmed=0 teleop=1 grippers=1
    for x in "$@"; do case $x in --confirmed) confirmed=1 ;; --no-teleop) teleop=0 ;; --no-grippers) grippers=0 ;;
        *) echo "unknown $x"; exit 1 ;; esac; done
    alive_pid "$RUN/real_control.pid" && { echo "real chain already running"; exit 1; }
    alive_pid "$RUN/sim_control.pid" && { echo "the sim chain is running: dwbc.sh stop first"; exit 1; }
    sonic_running && { echo "the SONIC chain is running (deploy / streamer / real_teleop.sh): stop it first"; exit 1; }
    local L=$DWBC_LOGS/real_$(date +%Y%m%d_%H%M%S); mkdir -p "$L"; ln -sfn "$L" "$DWBC_LOGS/real_latest"
    echo "$(date +%T) preflight (log dir $L)"
    check "$L" || { echo "!! preflight failed: see $L/check.json"; exit 1; }
    if [ $confirmed != 1 ]; then
        echo "Robot on the gantry, feet on the ground, rope slightly slack, area clear, someone at the e-stop?"
        read -r -p 'type YES to start (the robot WILL move): ' ans; [ "$ans" = YES ] || { echo aborted; exit 1; }
    fi
    cd /tmp
    echo $$ > "$RUN/real_start.pid"   # a start in progress: an estop now (even mid-ramp) ends in damping
    # grippers: Dex3 emulator (always: the control loop needs hand state) + Dex1 bridge for every streaming gripper
    local sides=""
    if [ $grippers = 1 ] && [ $DRY != 1 ]; then
        sides=$(bash "$HOME/piper_ws/real_teleop.sh" grippers 2>/dev/null | tail -1)
    fi
    # (dry run: the fake robot publishes sim Dex3 states itself)
    [ $DRY != 1 ] && start_bg real_dex3emu "$L/dex3_emu.log" python -u "$HERE/dex3_emu.py" --iface "$IFACE" --sides "${sides:-none}"
    if [ -n "$sides" ]; then
        [ -f /tmp/cdds_enp.xml ] || echo '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enp13s0"/></Interfaces></General></Domain></CycloneDDS>' > /tmp/cdds_enp.xml
        start_bg real_bridge "$L/bridge.log" env CYCLONEDDS_URI=file:///tmp/cdds_enp.xml PYTHONPATH=$SDK \
            $G1PY -u "$HOME/g1_bridge/sonic_dex1_bridge.py" --sides "$sides"
        echo "$(date +%T) Dex1 grippers: $sides (Dex3 emulator + bridge)"
    else
        echo "$(date +%T) no Dex1 gripper streaming: hands emulated as open, grippers not driven"
    fi
    # Unitree motion mode off, then a smooth ramp to the WBC default pose (robot motion starts here)
    if [ $DRY != 1 ]; then
        $LL release > "$L/release.log" 2>&1 || { echo "!! could not release the Unitree mode:"; cat "$L/release.log"; stop_aux; rm -f "$RUN/real_start.pid"; exit 1; }
    fi
    echo "$(date +%T) ramp to the WBC default pose (3 s)"; say "开始移动到默认姿势"
    $LL ramp --seconds 3 > "$L/ramp.log" 2>&1 || { echo "!! ramp failed:"; cat "$L/ramp.log"; $LL damp; stop_aux; rm -f "$RUN/real_start.pid"; exit 1; }
    tail -1 "$L/ramp.log"
    # control loop (real interface -> the 192.168.123.x NIC, DDS domain 0), then its guard right away
    # arm targets limited to DWBC_ARM_SPEED rad/s (upstream default 1000 = none): turning arm following on makes the
    # targets jump to the controllers; unlimited, the arms can pass upstream's 6 rad/s safety stop
    start_bg real_control "$L/control.log" python -u "$HERE/dwbc_run.py" control --interface real \
        --keyboard-dispatcher-type ros --no-enable-onscreen --upper-body-joint-speed "${DWBC_ARM_SPEED:-3.0}"
    start_bg real_guard "$L/guard.log" python -u "$HERE/dwbc_guard.py" --pidfile "$RUN/real_control.pid" \
        --flag "$FLAG" --iface "$IFACE" --csv "$L/thermal.csv" $([ "$QUIET" = 1 ] && echo --quiet)
    for i in $(seq 60); do
        grep -q "ROS keyboard dispatcher started" "$L/control.log" 2>/dev/null && break
        alive_pid "$RUN/real_control.pid" || break
        sleep 0.5
    done
    if ! grep -q "ROS keyboard dispatcher started" "$L/control.log"; then
        echo "!! control loop did not come up (the guard damps when it dies):"; tail -15 "$L/control.log"
        estop; exit 1
    fi
    echo "$(date +%T) control loop up: legs HOLD their angles, arms ramp to the start pose (2 s); leg policy OFF"
    if [ $teleop = 1 ]; then
        start_bg real_teleop "$L/teleop.log" python -u "$HERE/dwbc_run.py" teleop \
            --body-control-device pico --hand-control-device pico
        echo "$(date +%T) teleop loop (PICO) started"
    fi
    rm -f "$RUN/real_start.pid"
    say "全身控制已就绪"
    cat <<EOF
READY ($L)
  1. lower the gantry until the feet carry weight, rope STILL TAUT (the legs only hold angles now: with a slack rope
     the robot tips over before the policy runs; the dry run tipped 38 deg)
  2. leg policy on:  ~/piper_ws/dwbc/dwbc_real.sh key ']'   (or PICO menu + left trigger)
  3. then slacken the rope a little; the robot balances by itself
  4. arms follow the PICO controllers: PICO menu + right trigger (again = freeze arms)
  5. walk / turn: PICO sticks;  grippers: trigger = ~90 % closed, trigger + grip = fully closed
  stop: Mac app 急停 / physical e-stop / Unitree remote / ~/piper_ws/dwbc/dwbc_real.sh estop   (NOT the PICO)
EOF
}

stop_aux() {
    grp_kill "$RUN/real_teleop.pid"; grp_kill "$RUN/real_bridge.pid"; grp_kill "$RUN/real_dex3emu.pid"
}

estop() {
    # "active": something of ours may be commanding the motors (control loop, guard, or a start that got as far as
    # the release / ramp). Only then damping is sent: with nothing of ours running the robot may be under Unitree's
    # own controller, and a second lowcmd stream would fight it.
    local active=0 sp
    alive_pid "$RUN/real_control.pid" && active=1
    alive_pid "$RUN/real_guard.pid" && active=1
    if [ -f "$RUN/real_start.pid" ]; then
        active=1
        sp=$(cat "$RUN/real_start.pid")
        if kill -0 "$sp" 2>/dev/null && [ "$sp" != "$$" ]; then   # a start still running: stop it (and its ramp)
            local pg; pg=$(ps -o pgid= -p "$sp" | tr -d ' ')
            if [ -n "$pg" ] && [ "$pg" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then kill -KILL -- -"$pg" 2>/dev/null
            else kill -KILL "$sp" 2>/dev/null; pkill -KILL -P "$sp" 2>/dev/null; fi
            echo "stopped the start in progress (pid $sp)"
        fi
    fi
    if [ $active = 0 ]; then
        rm -f "$FLAG"; stop_aux
        echo "$(date +%T) dwbc: nothing of the real chain was running (no damping sent)"
        return 0
    fi
    if alive_pid "$RUN/real_guard.pid" && alive_pid "$RUN/real_control.pid"; then
        touch "$FLAG"   # the guard kills the control loop and damps (fastest path)
        for i in $(seq 40); do alive_pid "$RUN/real_guard.pid" || break; sleep 0.1; done
    fi
    if alive_pid "$RUN/real_control.pid" || alive_pid "$RUN/real_guard.pid" || [ -f "$RUN/real_start.pid" ]; then
        grp_kill "$RUN/real_guard.pid"
        [ -f "$RUN/real_control.pid" ] && kill -KILL -- -"$(cat "$RUN/real_control.pid")" 2>/dev/null
        $LL damp
    fi
    rm -f "$RUN/real_control.pid" "$RUN/real_guard.pid" "$RUN/real_start.pid" "$FLAG"
    stop_aux
    echo "$(date +%T) real chain stopped (motors in damping)"
    local L; L=$(readlink -f "$DWBC_LOGS/real_latest" 2>/dev/null) && [ -f "$L/guard.log" ] && tail -3 "$L/guard.log"
    return 0
}

status() {
    for n in real_control real_guard real_teleop real_dex3emu real_bridge; do
        if alive_pid "$RUN/$n.pid"; then echo "$n: running (pgid $(cat "$RUN/$n.pid"))"; else echo "$n: -"; fi
    done
    local L; L=$(readlink -f "$DWBC_LOGS/real_latest" 2>/dev/null) && echo "latest log dir: $L"
}

case "${1:-}" in
    check) L=$DWBC_LOGS/check_$(date +%Y%m%d_%H%M%S); mkdir -p "$L"; check "$L" ;;
    start) shift; start "$@" ;;
    key) shift; cd /tmp; python "$HERE/dwbc_key.py" "$@" ;;
    estop|stop) estop ;;
    status) status ;;
    *) sed -n '2,21p' "$0"; exit 1 ;;
esac
