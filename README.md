# Decoupled WBC（NVIDIA GR00T N1.5/N1.6 采集用的全身控制）部署

腿部用 RL 策略（Balance 和 Walk 两个 ONNX，按速度指令自动切换），手臂用 IK（pink + pinocchio）跟随 PICO 手柄。
上游代码在 `GR00T-WholeBodyControl/decoupled_wbc/`，没有改动。本目录是我们自己的启动、保护和验证脚本，
工作站上的副本在 `~/piper_ws/dwbc/`。搭建于 2026-11-17。

## 环境（工作站，不用 sudo，不用 Docker）

- conda 环境 `/data/Jiangzehua_Data/20261019_G1_WBC/envs/dwbc`：Python 3.10，加 RoboStack 的 ROS 2 Humble rclpy，再加 pip 依赖（CPU 版 torch、mujoco、pin、pink、onnxruntime 等）。
- `source ~/piper_ws/dwbc/dwbc_env.sh` 会做这些事：激活环境；把仓库和 SONIC 已编好的 XRoboToolkit SDK 加进 PYTHONPATH；把 conda 的 libstdc++ 放到最前；ROS 2 用私有 domain 77，并且只走本机回环。
- 官方 Docker 镜像要求当前用户在 docker 组里，这需要 sudo，所以没用。
- 两个策略 ONNX 是从 Mac 拷过去的，工作站上原来只有 git-lfs 占位文件。

## 仿真链路

```bash
~/piper_ws/dwbc/dwbc.sh sim            # 无界面；--gui 打开 MuJoCo 窗口；--pico 用 PICO 手柄遥操
~/piper_ws/dwbc/dwbc.sh sim-check      # 脚本代替操作员，端到端自检约 55 s，结果写进 sim_latest/check/
~/piper_ws/dwbc/dwbc.sh key ']'        # 腿部策略开；9 = 放开或挂上吊带
~/piper_ws/dwbc/dwbc.sh stop
```

- 仿真用 Unitree DDS 的 domain 7。另一个会话的 `fake_g1.py` 占着本机回环的 domain 0，`fake_g1_phys.py` 占着 domain 9。
- 自检的做法和遥操进程一样：在进程内跑同一套 IK，按 20 Hz 发送同样格式的目标消息。
- 2026-11-17 两次自检，8 项全部通过：
  - 站立：0.746 m，最大倾斜 8.6°，没有摔倒。
  - 手腕：动态跟随误差 0.8–1.4 cm，滞后 0.18 s。
  - 行走：指令 0.3 m/s，5 s 走了 1.07 m，停下后速度 0.03 m/s。
  - 转向：指令 0.5 rad/s，4 s 转了 89.5°。
  - 下蹲：0.746 → 0.618 m。
  - 夹爪：关节变化 1.31 rad。
- **手臂会因重力下垂 3–4 cm**。上游的 `enable_gravity_compensation` 只是记录下来，控制循环发出的前馈力矩一直是 0。

## 真机链路

**平时从 Mac 遥操台 app 启动**（g1_hub 0.4，2026-11-17）：

1. 「检查」通过后点「待命」。
2. 在「控制器」里选 **Decoupled WBC**，勾满四项后发送。hub 会执行 `dwbc_real.sh start --confirmed`，阶段依次是：启动中 → 就绪。
3. 就绪后，阶段卡片上会出现「开启腿部平衡 / 关闭腿部平衡」两个按钮，分别发送 `]` 和 `o`。
4. app 的急停也能停这条链路：`estop_direct.sh` 会**先**执行 `dwbc_real.sh estop`，也就是阻尼，再执行 SONIC 的急停。
5. 控制器可以随时换回 SONIC，但两条链路不能同时运行，hub 会拒绝。

下面这些命令是同一套操作的命令行版本：

```bash
~/piper_ws/dwbc/dwbc_real.sh check                  # 只读预检，不给机器人发任何指令
~/piper_ws/dwbc/dwbc_real.sh start --confirmed      # 机器人会动！确认吊架、场地和急停有人守着之后才执行
~/piper_ws/dwbc/dwbc_real.sh key ']'                # 腿部策略开
~/piper_ws/dwbc/dwbc_real.sh estop                  # 杀掉控制循环，全身阻尼（kp 0 / kd 8）
~/piper_ws/dwbc/dwbc_real.sh status
```

`start` 依次做这些事：
1. 预检：lowstate、躯干 IMU（`rt/secondary_imu`）、mode_machine=5、电机温度。
2. 夹爪：Dex3 手部模拟器加 Dex1 桥接；夹爪服务没起来时，用 `real_teleop.sh grippers` 拉起。
3. 释放宇树运动模式。
4. 3 s 平滑过渡到 WBC 默认站姿。
5. 启动控制循环，同时启动守护进程。
6. 启动 PICO 遥操进程。

**真机操作顺序**（dry run 里试出来的）：

1. `start` 之后，腿只是保持角度，还没有在平衡。这时把龙门架放低，**放到脚吃上力、吊绳还绷着**。
2. 打开腿部策略：执行 `key ']'`，或者按 PICO 的菜单键 + 左扳机。
3. **然后再放松吊绳**。如果先放松吊绳，机器人在策略启动前就会倒，dry run 里倒了 38°。
4. 手臂跟随：PICO 菜单键 + 右扳机，再按一次冻结手臂。
5. 摇杆控制行走和转向。
6. 夹爪：只按扳机约闭合 90%，扳机加侧握键为全闭。

**怎么停**：
- 能用：Mac app 的急停（hub 0.4 起）、物理急停、宇树遥控器、`dwbc_real.sh estop`。
- **不能用：PICO**，它上面没有急停。
- `o` 键（「关闭腿部平衡」）只是关掉腿部策略，腿会**保持角度**，不是急停。

**手臂限速**：手臂目标的关节速度上限是 3 rad/s，可以用 `DWBC_ARM_SPEED` 调整；上游默认 1000，等于不限速。
- 原因：打开手臂跟随时，目标会直接跳到手柄对应的位置，不限速的话手臂可能超过上游 6 rad/s 的安全上限而停机。

**PICO 键位**（上游 `pico_streamer.py`）：
- 菜单键 + 左扳机：开 / 关腿部平衡。
- 菜单键 + 右扳机：开 / 关手臂跟随，打开时立即生效，没有倒计时。
- 左摇杆：前后、横移，最高 0.5 m/s。右摇杆左右：转向，最高 1 rad/s。
- Y / X：升高 / 降低身体，按住约 0.2 m/s，范围 0.2–0.74 m。
- 扳机：夹爪约合 90%；扳机 + 侧握键：全闭。
- A / B 只在跑上游数据导出时有用（开始录制 / 丢弃），现在不用。

**守护进程 `dwbc_guard.py`**：下面三种情况会杀掉控制循环并发阻尼：
- 控制循环退出。上游遇到手臂速度超过 6 rad/s 会直接 `sys.exit`，不会自己发阻尼。
- 执行了 `estop`。
- 电机过热：外壳 85 °C 或绕组 115 °C 持续 2 s。到 75 / 100 °C 时会先语音提醒。

**Dex1 夹爪**：`dex3_emu.py` 用 Dex1 的位置伪装出 Dex3 的手部状态。拇指固定在 60°，这样上游的拇指标定会得到 0 偏差。没有这层伪装，控制循环会因为读不到手部状态直接崩溃。夹爪指令沿用 SONIC 的 `sonic_dex1_bridge.py`。

## 不接机器人的验证

```bash
unshare -rn bash ~/piper_ws/dwbc/dwbc_dryrun.sh       # 真机代码路径对假机器人，约 2 min
```

- 原理：在私有网络命名空间里，给回环接口加上 192.168.123.99，让 `--interface real` 走真机代码路径。
- `dwbc_fake_robot.py` 是一个独立的 MuJoCo 进程，扮演机器人，带可以放低、放松的龙门架。
- 2026-11-17 全部通过：
  - 启动：过渡结束时误差 0.085 rad。
  - 站立：放低后开策略，再放松吊绳，站稳在 0.747 m，倾斜 0.5°。
  - 行走：按键走了 1.45 m 后停住。
  - `estop`、`kill -9` 控制循环、过热三种情况都切到了阻尼。
- 夹爪链路单测：在私有命名空间里先启动 `dex3_emu.py --iface lo`，再跑 `dwbc_grip_test.py`。

## 上线流程：改动上真机前必须过仿真门禁（2026-11-27 起）

手臂逆解的任何改动（参数、`dwbc_env.sh` 里的 `DWBC_IK_*`、`dwbc_ik_global.py` 代码）上真机前，按这个顺序走：

1. **仿真门禁**：和**现在真机的配置**比，用全部录下的操作员输入（`tests/ik_bench.py list`，最新的录像也算），带上半身回放走路：
   ```bash
   source dwbc_env.sh
   python tests/predeploy.py NAME --env DWBC_IK_ELBOW_MAX=1.0          # 改 env 变量
   python tests/predeploy.py NAME --set NEW_KNOB=0.5 --ik my_solver.py  # 改代码 / 新参数
   python tests/predeploy.py NAME ... --target jumps                     # 想改善的维度（至少 +5 分）
   ```
   门禁 v2（`tests/sim_eval.py`）：手臂突变三项计数都不许增加；逆解 CPU 时间 p99 ≤ 50 ms（卡顿）；带上半身走路最多掉 5 分、不许摔；腰只设上限；抖动 / 手跟随最多掉 5 分。
   **FAIL 不上真机。** 报告在 `/data/robot_data/g1/ik_bench/results/predeploy/`。
2. **人审**：看报告（PASS 的理由、每个维度的分数），决定是否上真机。
3. **真机 A/B**：开关先放在 `dwbc_env.sh` 的环境变量里，默认关；操作员做和问题相关的动作，录 episode。
4. **评分卡确认**：app 监控 › 运行评分，和上一次运行对比；没变差再把默认值改成开。
5. 每次真机运行后，把新的 episode 加进 `ik_bench`（`python tests/ik_bench.py add`），下一个改动就要过这批新数据。

不在门禁里的：腿部策略、增益、网络等改动（用 `tests/sim_eval.py walk` 和真机评分卡）。大模型提的方案同样要走这个流程，没有例外。

## 还没验证的

- 真机上电后的整条链路：释放模式、站立、PICO 遥操、Dex1 夹爪。2026-11-17 只做了只读预检，全部通过：lowstate 和躯干 IMU 都是 1035 Hz，mode_machine 5，两侧 Dex1 501 Hz，当时处在宇树 ai 模式。
- 实际戴着 PICO 的遥操：遥操进程能连上 XRoboToolkit 服务，但头显当时没有数据。
- 数据导出 `run_g1_data_exporter.py`：需要 lerobot，还没装。

## 控制链路延迟监控（2026-11-20）

`dwbc_latency.py` 在 teleop 和 control 两个进程里给每一段打时间戳，默认开启（`DWBC_LATENCY=0` 关闭）。它只计时，不改变两个循环算出的任何结果；teleop 发出的目标消息里会多一个 `lat` 字段（时间戳），control 进程不读它。监控代码出错时会打印一行并自动关闭，不会影响控制。

每次真机运行，下面三个文件写进那次的日志目录 `dwbc_logs/real_<时间>/`：

- `lat_teleop.csv`：teleop 每一帧一行
- `lat_control.csv`：control 每取走一个目标一行，[1]–[10] 全在这一行里
- `lat_track.csv`：control 每圈一行，记录手臂 14 个关节的指令值和实测值，用来算 [11]

```bash
python3 ~/piper_ws/dwbc/dwbc_latency.py watch          # 实时，每秒刷新，统计最近 5 s
python3 ~/piper_ws/dwbc/dwbc_latency.py report [目录]   # 整场汇总，同时写 lat_summary.txt
ssh -t workstation-lan python3 ~/piper_ws/dwbc/dwbc_latency.py watch   # 在 Mac 上看
```

| 段 | 含义 | 量法 |
| --- | --- | --- |
| [1] PICO 采样→teleop 读到 | Wi‑Fi 加上数据在服务里等待被读的时间 | 工作站时钟减去 PICO 时间戳。两边时钟没对齐，所以显示的是超出整场最小值的部分；原始差值另外列出 |
| [2] teleop 周期 | 两次读手柄的间隔，目标 50 ms | — |
| [3] 读手柄+位姿处理 | `TeleopStreamer.get_streamer_data` | — |
| [4] 手臂 IK | `TeleopRetargetingIK.get_action`；`DWBC_IK_GLOBAL=1` 时走多起点 IK | — |
| [5] teleop 其它 | 激活判断、A+B+X+Y 回站姿等 | — |
| [6-7] 打包+ROS 2 传输 | msgpack 打包、发布，到 control 的回调收到 | — |
| [8] 等控制环取走 | 消息到达后，等 50 Hz 循环来取 | — |
| [9] 控制环计算→lowcmd | 取走后到 `rt/lowcmd` 写出（腿部策略、安全检查） | — |
| [10] 插值到位 | 取走后到插值轨迹到达这一帧目标的时刻 | 至少是发布时刻 +50 ms；手臂速度上限 `DWBC_ARM_SPEED`（3 rad/s）起作用时更长 |
| [11] 机器人跟随滞后 | 实测关节相对指令关节的滞后 | 互相关；只算 5 s 内动了 3° 以上的关节，先去掉均值（重力下垂是偏差，不算滞后） |

最后的“估算 PICO 动作→机器人到位”= [1] + 工作站内合计（读到→指令到位）+ [11]。所有统计只算手臂在跟随的帧，也就是 A+X 打开之后。

**仿真自测**（不需要机器人和头显）：`unshare -rn bash ~/piper_ws/dwbc/tests/lat_sim_test.sh 30`。脚本在独立的网络命名空间里运行，ROS 用 domain 78，用假 PICO 驱动手臂，碰不到真机链路。

2026-11-20 的自测结果（MuJoCo 和多起点 IK 同时在跑）：工作站内合计 p50 136 ms；其中 IK p50 36 ms、p90 146 ms，插值 p50 46 ms。
