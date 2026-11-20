#!/usr/bin/env bash
# NVIDIA Decoupled WBC (GR00T N1.5/N1.6 data-collection controller: RL legs + IK arms) on the workstation.
#
#   ~/piper_ws/dwbc/dwbc.sh sim [--gui] [--pico]   MuJoCo sim on lo, Unitree DDS domain 7 (another session's fake_g1
#                                                  owns lo/domain 0). --gui: MuJoCo viewer; --pico: teleop loop with
#                                                  the PICO controllers (XRoboToolkit), else drive it with sim-check
#   ~/piper_ws/dwbc/dwbc.sh sim-check              scripted end-to-end check against a running sim (dwbc_sim_check.py)
#   ~/piper_ws/dwbc/dwbc.sh key K ...              keys to the control loop: ']' policy on, 'o' off, 9 band (sim)
#   ~/piper_ws/dwbc/dwbc.sh stop                   stop the sim chain
#   ~/piper_ws/dwbc/dwbc.sh status
# Real robot: dwbc_real.sh (separate script, own checks and e-stop).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/dwbc_env.sh"     # before set -u: conda's ROS activate scripts read unset variables
set -u
RUN=$HOME/piper_ws/dwbc/run          # pid files
mkdir -p "$RUN" "$DWBC_LOGS"
SIM_DOMAIN=7

alive_pid() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
grp_kill() {  # SIGINT, then SIGTERM, then SIGKILL to the process group in pid file $1
    local f=$1 p; [ -f "$f" ] || return 0; p=$(cat "$f")
    for sig in INT TERM KILL; do
        kill -0 "$p" 2>/dev/null || break
        kill -s $sig -- -"$p" 2>/dev/null
        for i in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
    done
    rm -f "$f"
}
start_bg() {  # name logfile cmd...   (own session -> own process group; pid file = group id)
    local name=$1 log=$2; shift 2
    setsid bash -c 'echo $$ > "'"$RUN/$name.pid"'"; exec "$@"' _ "$@" > "$log" 2>&1 < /dev/null &
    for i in $(seq 20); do [ -f "$RUN/$name.pid" ] && break; sleep 0.05; done
}
real_running() { alive_pid "$RUN/real_control.pid"; }

sim() {
    local gui=0 pico=0
    for a in "$@"; do case $a in --gui) gui=1 ;; --pico) pico=1 ;; *) echo "unknown $a"; exit 1 ;; esac; done
    real_running && { echo "the REAL chain is running (dwbc_real.sh): stop it first"; exit 1; }
    alive_pid "$RUN/sim_control.pid" && { echo "sim already running (dwbc.sh stop first)"; exit 1; }
    local L=$DWBC_LOGS/sim_$(date +%Y%m%d_%H%M%S); mkdir -p "$L"; ln -sfn "$L" "$DWBC_LOGS/sim_latest"
    local view=--no-enable-onscreen; [ $gui = 1 ] && view=--enable-onscreen
    [ $gui = 1 ] && export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
    cd /tmp
    DWBC_DDS_DOMAIN=$SIM_DOMAIN start_bg sim_control "$L/control.log" \
        python -u "$HERE/dwbc_run.py" control --interface sim --keyboard-dispatcher-type ros $view
    for i in $(seq 60); do grep -q "ROS keyboard dispatcher started" "$L/control.log" 2>/dev/null && break; sleep 0.5; done
    grep -q "ROS keyboard dispatcher started" "$L/control.log" || { echo "control loop did not come up:"; tail -20 "$L/control.log"; grp_kill "$RUN/sim_control.pid"; exit 1; }
    echo "$(date +%T) sim control loop up (domain $SIM_DOMAIN, log $L/control.log)"
    if [ $pico = 1 ]; then
        start_bg sim_teleop "$L/teleop.log" python -u "$HERE/dwbc_run.py" teleop \
            --body-control-device pico --hand-control-device pico
        echo "$(date +%T) teleop loop (PICO) started: log $L/teleop.log"
        echo "PICO: menu+left trigger = legs policy on/off, menu+right trigger = arm following on/off, sticks = walk/turn"
        echo "sim: '~/piper_ws/dwbc/dwbc.sh key 9' releases the elastic band after the legs policy is on"
    fi
}

stop_sim() {
    grp_kill "$RUN/sim_teleop.pid"; grp_kill "$RUN/sim_check.pid"; grp_kill "$RUN/sim_control.pid"
    echo "$(date +%T) sim chain stopped"
}

status() {
    for n in sim_control sim_teleop real_control real_teleop real_dex3emu real_bridge; do
        if alive_pid "$RUN/$n.pid"; then echo "$n: running (pgid $(cat "$RUN/$n.pid"))"; else echo "$n: -"; fi
    done
}

case "${1:-}" in
    sim) shift; sim "$@" ;;
    sim-check) shift; cd /tmp; L=$(readlink -f "$DWBC_LOGS/sim_latest"); python -u "$HERE/dwbc_sim_check.py" --out "$L/check" "$@" ;;
    key) shift; cd /tmp; python "$HERE/dwbc_key.py" "$@" ;;
    stop) stop_sim ;;
    status) status ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
