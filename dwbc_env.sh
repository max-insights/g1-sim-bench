# Environment for NVIDIA Decoupled WBC (GR00T-WholeBodyControl/decoupled_wbc) on the workstation.
#   source ~/piper_ws/dwbc/dwbc_env.sh
# conda env (no sudo, no Docker): python 3.10 + ROS 2 Humble rclpy from RoboStack + pip deps, built 2026-11-17.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /data/Jiangzehua_Data/20261019_G1_WBC/envs/dwbc
export DWBC_REPO=$HOME/piper_ws/GR00T-WholeBodyControl
# conda's libstdc++ first: a pip wheel imported before torch otherwise pulls the older system one (CXXABI_1.3.15 error)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# decoupled_wbc is used from the repo (its pyproject cannot be pip-installed: readme sits outside the package)
# + the XRoboToolkit pybind SDK already built for SONIC's .venv_teleop (cp310)
export PYTHONPATH=$DWBC_REPO:$DWBC_REPO/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64${PYTHONPATH:+:$PYTHONPATH}
# ROS 2 only links our own processes: private domain, loopback only, so nothing reaches the robot's DDS on enp13s0.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export DWBC_LOGS=$HOME/piper_ws/dwbc_logs
# waist in the arm IK -> the leg policy leans the torso (upstream --enable-waist); DWBC_ENABLE_WAIST=0 turns it off
export DWBC_ENABLE_WAIST=${DWBC_ENABLE_WAIST:-1}
# arm IK posture weight (upstream 0.01 gets stuck: pinned elbow / backward lean); unset it to go back to upstream
export DWBC_IK_POSTURE_COST=${DWBC_IK_POSTURE_COST:-0.01}
# multi-start (global) arm IK, dwbc_ik_global.py (2026-11-18); DWBC_IK_GLOBAL=0 turns it off
export DWBC_IK_GLOBAL=${DWBC_IK_GLOBAL:-1}
# arm IK smoothing v2 (2026-11-24: speed bound inside the IK, singularity damping; offline replay: joint jerks
# 17 -> 2 per minute, hand offset unchanged). 0 = the IK as before. Takes effect at the next DWBC start.
export DWBC_IK_SMOOTH=${DWBC_IK_SMOOTH:-0}
# A+B+X+Y: leg policy on, and after the first press a slow return to standing (2026-11-18); 0 turns it off
export DWBC_ABXY_RESET=${DWBC_ABXY_RESET:-1}
# single-step mode (2026-11-23, dwbc_step_ext.py): PICO left stick click toggles it, a flick = one step;
# keys t/g/f/h on the control loop. 0 turns it off
export DWBC_STEP=${DWBC_STEP:-1}
# per-direction tuning: speed m/s, stop at the Nth lift|touchdown (defaults in dwbc_step.py PROFILES):
# export DWBC_STEP_FWD=0.20,lift,2   DWBC_STEP_BACK=0.15,lift,2   DWBC_STEP_LEFT=0.20,lift,2   DWBC_STEP_RIGHT=0.20,lift,2
# one math-library thread per process (2026-11-23). Unlimited, torch / numpy / OpenBLAS start a thread per core in both
# loops and they fight (load ~40, control 10 cores, teleop 8). Sim A/B (tests/thread_ab.sh): arm IK p50 38 -> 5.5 ms,
# p90 160 -> 8, stalls 10% -> 0, control loop missed-timing reports 75-328 -> 0. Set e.g. OMP_NUM_THREADS=24 to undo.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
