"""Multi-start ("global") arm IK for the decoupled-WBC teleop loop (2026-11-17).

Upstream BodyIKSolver is a pink differential IK: every teleop frame it takes 3 small QP steps from the previous
solution. That is purely local, so once the arms reach an awkward branch (an elbow pinned at its limit with the wrist
centimetres off, or - with the waist in the IK - a backward lean with the arms ~100 deg away from the start) nothing
ever pulls them out.

This solver keeps the upstream solve as the *active* solution and runs a few extra candidate solutions next to it:
  * every candidate is tracked with the same tasks each frame (warm start, same steps as the active one);
  * round robin, one candidate is re-seeded every RESEED_EVERY frames from a fixed seed set (home pose, two reach
    poses) or from a random perturbation of the active solution, and is converged with RESEED_STEPS steps;
  * each frame all solutions are scored: task cost = wrist position error (cm) + rotation error (20 deg = 1) + joints
    sitting on a limit; posture cost = distance from the home posture. A candidate takes over only when, for SUSTAIN
    frames in a row, it tracks clearly better (TASK_MARGIN) or tracks as well and sits much closer to the home posture
    (POSTURE_MARGIN, e.g. the backward-lean branch); switches are at least MIN_DWELL_S apart and may retarget a
    running blend (it restarts from the current output, so the output stays continuous);
  * a switch never jumps: the published joints blend from the old output to the new solution over
    max(BLEND_MIN_S, largest joint change / BLEND_RAD_PER_S) seconds (the control loop shuts down above
    6 rad/s arm joint speed), with a smoothstep profile.
The published joints never move faster than VMAX rad/s (the control loop shuts down above 6 rad/s for arm joints;
a stalled teleop frame otherwise makes the target, and the arm, jump).
It also limits the IK waist pitch to >= WAIST_PITCH_MIN (no backward lean from the arm IK).

Null-space posture (2026-11-18): upstream's posture task is a soft cost next to the hand tasks, so at 0.01 it pulls
the arms toward the home posture only very slowly (hand still, arm creeping up to ~30 deg) and at 0.1 it costs ~1 cm
of hand accuracy. After every QP step this solver adds a second step that moves the arm / waist joints toward the home
posture only inside the null space of both hand tasks (damped pseudo-inverse of the stacked 6D hand Jacobians):
the hands do not feel it, the redundant degrees of freedom settle within ~1 s. It fades out when a hand is more than
NULL_ERR_FULL_CM off its target (edge of reach). NULL_GAIN = 0 turns it off.
When the target stands still and a frame of steps brings the active solution no closer (out of reach), that frame's
steps are dropped (HOLD_NO_PROGRESS), so the arm does not creep along the reach boundary.

Enable with DWBC_IK_GLOBAL=1 (dwbc_run.py patches teleop_retargeting_ik.BodyIKSolver with this class).
"""
import time

import numpy as np
import pink
from pink import solve_ik

from decoupled_wbc.control.teleop.solver.body.body_ik_solver import BodyIKSolver, WeightedPostureTask


class GlobalBodyIKSolver(BodyIKSolver):
    N_CAND = 4
    RESEED_EVERY = 2          # frames between two re-seeds (each re-seeds one candidate)
    RESEED_STEPS = 15
    TASK_MARGIN = 1.5         # task-cost units (1 = 1 cm of wrist position error)
    POSTURE_MARGIN = 1.0      # posture-cost units, with the task cost no worse than TASK_SLACK
    TASK_SLACK = 0.3
    SUSTAIN = 3               # frames a candidate must stay better before it takes over
    MIN_DWELL_S = 0.3         # minimum time between two switches (a switch may retarget a running blend)
    BLEND_MIN_S = 0.6
    BLEND_RAD_PER_S = 2.0
    VMAX = 3.0                # rad/s cap on every published joint (below the control loop's 3.5 rad/s interpolation cap)
    WAIST_PITCH_MIN = -0.087  # rad (-5 deg)
    LIMIT_EPS = 0.02          # rad: "sitting on a limit"
    POSTURE_W = 0.6           # posture cost per rad of (weighted) distance from the home posture
    NULL_GAIN = 2.0           # 1/s (IK time): null-space pull toward the home posture
    NULL_VMAX = 1.0           # rad/s (IK time) cap on the null-space step
    NULL_DAMP = 0.02          # damping of the hand-Jacobian pseudo-inverse
    # full null-space pull while both hands are within NULL_ERR_FULL_CM of their targets, none from NULL_ERR_OFF_CM.
    # None = by IK mode (2026-11-18 sweep, max joint drift with the hands still): waist in the IK 0.3 / 1.0 cm
    # (1.3 deg; wider gates let the pull creep the stretched arm), arms only 1.0 / 3.0 cm (1.7 deg; arms-only
    # holds often sit 0.3-1 cm off at the reach limit and still need the pull)
    NULL_ERR_FULL_CM = None
    NULL_ERR_OFF_CM = None
    # waist priority (2026-11-18, operator: "waist last out, first back"):
    #  last out  - WAIST_DAMP: an extra QP task that penalises waist motion each step (target = current waist), so
    #              the arms take a reach first and the waist only moves for what the arms cannot do; 0 = off
    #  first back - the null-space pull acts on the waist with WAIST_NULL_MULT x the gain and its own, wider hand-error
    #              gate (WAIST_GATE_FULL_CM / WAIST_GATE_OFF_CM), so the waist straightens up while the hands are still
    #              moving instead of only once they stand still
    WAIST_DAMP = 0.0
    # weight of the waist in the posture terms (null-space pull + candidate scoring) relative to one arm joint. At 3 the
    # pull toward the home posture is cheaper to satisfy by bending the waist than by moving 14 arm joints, so a 15 cm
    # reach already bent the waist ~27 deg and it stayed bent after the hands came back
    WAIST_POSTURE_W = 3.0
    WAIST_NULL_MULT = 1.0
    WAIST_GATE_FULL_CM = None   # None = same gate as the arms
    WAIST_GATE_OFF_CM = None
    # waist priority by mode (2026-11-18, operator: "waist last out, first back"). Soft weights alone did not work:
    # bending the waist moves both hands at once, so the QP and the null-space pull always preferred it.
    #  locked    - the waist is held at home while the arms alone reach the targets ("last out");
    #  free      - entered when the locked solution stays off target (task error > WAIST_UNLOCK_ERR for
    #              WAIST_UNLOCK_FRAMES frames): the waist joins the IK as before;
    #  returning - entered from free once an always-running upright tracker (waist locked at home) reaches the
    #              targets (error < WAIST_RELOCK_ERR for WAIST_RELOCK_FRAMES frames): the waist reference ramps back
    #              to home at WAIST_BACK_RAD_PER_S while the arms keep the hands on target ("first back"). The ramp
    #              pauses while the hands are off by more than WAIST_RETURN_OK_ERR; it falls back to free when the error
    #              stays above WAIST_UNLOCK_ERR; at home the waist is locked again. No solution switch, no joint blend.
    WAIST_PRIORITY = True
    WAIST_UNLOCK_ERR = 2.0
    WAIST_UNLOCK_FRAMES = 6
    WAIST_RELOCK_ERR = 0.8
    WAIST_RELOCK_FRAMES = 10         # real run 19:03-19:04: 5 frames flapped free <-> returning near the reach edge
    WAIST_RETRY_S = 2.0              # after a straightening attempt fails, wait this long before the next one
    WAIST_RETURN_OK_ERR = 1.0
    WAIST_BACK_RAD_PER_S = 0.25     # waist straightening speed (rad/s, ~14 deg/s)
    HOLD_NO_PROGRESS = True   # target still + active solution off target + a frame of steps gains nothing: keep q
    HOLD_ERR = 0.3            # task-error units (cm + deg/20)
    HOLD_MIN_GAIN = 0.02
    # output smoothing (2026-11-24, branch ik-reach-fix; offline replay tests/ik_replay.py of the 11-24 runs): the
    # published joints follow the IK with a bounded acceleration instead of a bare VMAX clip. A one-frame IK jump
    # (waist lock / unlock, near-singular arm, re-convergence) used to come out as a full-VMAX step that started and
    # stopped within one frame; now it ramps up and brakes so it stops on the target without overshoot. 0 = off
    # (bare VMAX clip, as before).
    OUT_AMAX = 15.0           # rad/s^2 (replay sweep 10-40: 15 cut jerk events 274 -> 11, adds no lag)
    # relative switch margin: a candidate must also beat the active solution's task cost by this fraction. With the
    # hands far out of reach every candidate is poor and an absolute 1.5 cm margin switched on noise (run 11-24 16:15
    # after the e-stop: 30 switches of 20-230 deg while the targets were metres away). 0 = absolute margin only.
    TASK_MARGIN_REL = 0.2
    clock = staticmethod(time.monotonic)   # benchmarks replace it with a simulated clock
    verbose = False                        # dwbc_run.py: print every switch to teleop.log

    def initialize(self):
        super().initialize()
        model = self.configuration.model
        idx = self.robot.joint_to_dof_index
        self.lo = np.array(model.lowerPositionLimit, dtype=float).copy()
        self.hi = np.array(model.upperPositionLimit, dtype=float).copy()
        wp = idx.get("waist_pitch_joint")
        if wp is not None:
            self.lo[wp] = max(self.lo[wp], self.WAIST_PITCH_MIN)
            model.lowerPositionLimit[wp] = self.lo[wp]
        self.q_home = np.clip(self.configuration.q.copy(), self.lo, self.hi)
        self.configuration.update(self.q_home.copy())
        # posture weights for the cost: waist counts 3x (a leaning torso moves the whole robot)
        # finger joints are in the IK vector too but are driven by the hand IK: keep them out of the cost / seeds
        self.pw = np.ones(len(self.q_home))
        self.armwaist = np.zeros(len(self.q_home), dtype=bool)
        for name, i in idx.items():
            if "hand" in name:
                self.pw[i] = 0.0
            else:
                self.armwaist[i] = True
            if name.startswith("waist"):
                self.pw[i] = self.WAIST_POSTURE_W
        self.seeds = [self.q_home.copy()]
        for sp, el in ((-0.5, 0.9), (-1.1, 0.3)):   # elbow bent in front / arms reaching forward
            s = self.q_home.copy()
            for side in ("left", "right"):
                for j, v in ((f"{side}_shoulder_pitch_joint", sp), (f"{side}_elbow_joint", el)):
                    if j in idx:
                        s[idx[j]] = v
            self.seeds.append(np.clip(s, self.lo, self.hi))
        self.cands = [pink.Configuration(model, model.createData(), self.seeds[k % len(self.seeds)].copy())
                      for k in range(self.N_CAND)]
        self.better = [0] * self.N_CAND
        self.hand_frames = [self.robot.supplemental_info.hand_frame_names[s] for s in ("left", "right")]
        self.frame_n = 0
        self.reseed_k = 0
        self.seed_k = 0
        self.blend = None          # (t0, duration, q_from)
        self.no_switch_until = -1e9
        self.q_out = self.q_home.copy()
        self.v_out = np.zeros(len(self.q_home))   # published joint velocity (OUT_AMAX)
        self.t_last = None
        self.rng = np.random.default_rng(0)
        self.switches = 0
        self.last_switch = None    # (time, reason, task/posture costs, max joint change)
        self.switch_log = []       # (time, max joint change rad, blend duration s, reason) for metrics
        self._fast = False         # set by the warm-up patch below
        self._tgt_prev = None
        self.is_waist = np.zeros(len(self.q_home), dtype=bool)
        for name, i in idx.items():
            if name.startswith("waist"):
                self.is_waist[i] = True
        self._waist_task = None
        if self.WAIST_DAMP > 0 and self.is_waist.any():
            self._waist_task = WeightedPostureTask(cost=self.WAIST_DAMP, weights=self.is_waist.astype(float),
                                                   lm_damping=0.0)
        waist_in_ik = any(n.startswith("waist") for n in idx)
        prio = bool(self.WAIST_PRIORITY and self.is_waist.any())
        self.waist_mode = "locked" if prio else "free"
        self.waist_ref = self.q_home[self.is_waist].copy()
        self.upright = pink.Configuration(model, model.createData(), self.q_home.copy()) if prio else None
        self.unlock_n = 0
        self.relock_n = 0
        self._upright_k = 0
        self._retry_after = -1e9
        self.waist_log = []        # (time, "unlocked" / "relocked", task error)
        self.null_full = self.NULL_ERR_FULL_CM if self.NULL_ERR_FULL_CM is not None else (0.3 if waist_in_ik else 1.0)
        self.null_off = self.NULL_ERR_OFF_CM if self.NULL_ERR_OFF_CM is not None else (1.0 if waist_in_ik else 3.0)

    # ------------------------------------------------------------------ helpers
    def _limits(self, lock):
        if lock is None:
            return self.lo, self.hi
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[self.is_waist] = lock - 1e-6
        hi[self.is_waist] = lock + 1e-6
        return lo, hi

    def _step(self, cfg, n, lock=None):
        """n IK steps; lock = waist joint values to hold the waist at (None = waist free)"""
        locked = lock is not None
        lo, hi = self._limits(lock)
        model = cfg.model
        if locked:      # the QP must know the waist is fixed, else it plans waist motion that the clip then drops
            model.lowerPositionLimit[:] = lo
            model.upperPositionLimit[:] = hi
            cfg.update(np.clip(cfg.q, lo, hi))
        try:
            for _ in range(n):
                if self.NULL_GAIN > 0:      # null-space posture first: the QP step right after corrects its leakage
                    self._null_step(cfg, lo, hi, locked)
                tasks = list(self.tasks.values())
                if self._waist_task is not None and not locked:
                    self._waist_task.set_target_from_configuration(cfg)
                    tasks.append(self._waist_task)
                v = solve_ik(cfg, tasks, dt=self.dt, solver=self.solver)
                cfg.update(np.clip(cfg.q + v * self.dt * self.amplify_factor, lo, hi))
        finally:
            if locked:
                model.lowerPositionLimit[:] = self.lo
                model.upperPositionLimit[:] = self.hi

    def _null_step(self, cfg, lo, hi, locked=False):
        """move toward the home posture inside the null space of both hand tasks"""
        # only when the hands sit on their targets: at the edge of reach (arm stretched, Jacobian near singular) the
        # null space is ill-defined and the posture pull would fight the hand task and creep the elbow
        e = 0.0
        for f in self.hand_frames:
            tgt = self.tasks[f].transform_target_to_world
            if tgt is not None:
                e = max(e, 100.0 * float(np.linalg.norm(cfg.get_transform_frame_to_world(f).translation
                                                        - tgt.translation)))
        gate = float(np.clip((self.null_off - e) / (self.null_off - self.null_full), 0.0, 1.0))
        wf = self.WAIST_GATE_FULL_CM if self.WAIST_GATE_FULL_CM is not None else self.null_full
        wo = self.WAIST_GATE_OFF_CM if self.WAIST_GATE_OFF_CM is not None else self.null_off
        wgate = float(np.clip((wo - e) / (wo - wf), 0.0, 1.0)) * self.WAIST_NULL_MULT
        if gate <= 0.0 and wgate <= 0.0:
            return
        J = np.vstack([self.tasks[f].compute_jacobian(cfg) for f in self.hand_frames])
        g = np.where(self.is_waist, wgate, gate)
        v = g * self.NULL_GAIN * self.pw * (self.q_home - cfg.q)
        if locked:                  # waist fixed: null space of the arms only
            J = J.copy()
            J[:, self.is_waist] = 0.0
            v[self.is_waist] = 0.0
        JJt = J @ J.T + (self.NULL_DAMP ** 2) * np.eye(J.shape[0])
        v = v - J.T @ np.linalg.solve(JJt, J @ v)
        if locked:
            v[self.is_waist] = 0.0
        m = float(np.abs(v).max())
        if m > self.NULL_VMAX:
            v *= self.NULL_VMAX / m
        cfg.update(np.clip(cfg.q + v * self.dt, lo, hi))

    def _waist_mode(self, mode, now, err, why):
        self.waist_mode, self.unlock_n, self.relock_n = mode, 0, 0
        self.waist_log.append((now, mode, err))
        if self.verbose:
            print(f"[dwbc_ik] waist {mode}: {why}", flush=True)

    def _task_err(self, cfg):
        """wrist position error (cm) + rotation error (20 deg = 1), summed over both hands"""
        c = 0.0
        for f in self.hand_frames:
            tgt = self.tasks[f].transform_target_to_world
            if tgt is None:
                continue
            T = cfg.get_transform_frame_to_world(f)
            c += 100.0 * float(np.linalg.norm(T.translation - tgt.translation))
            Rr = T.rotation.T @ tgt.rotation
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0))))
            c += 0.05 * ang
        return c

    def _cost(self, cfg):
        """(task cost, posture cost)"""
        c = self._task_err(cfg)
        q = cfg.q
        margin = np.minimum(q - self.lo, self.hi - q)
        c += 0.5 * float(np.sum((margin < self.LIMIT_EPS) & self.armwaist))
        return c, self.POSTURE_W * float(np.linalg.norm((q - self.q_home) * self.pw))

    # ------------------------------------------------------------------ main
    def __call__(self, target_pose):
        import pinocchio as pin

        for link_name, pose in target_pose.items():
            if link_name in self.tasks:
                self.tasks[link_name].set_target(pin.SE3(pose[:3, :3], pose[:3, 3]))

        now = self.clock()
        if self._fast:
            # IK warm-up (TeleopRetargetingIK runs 50 calls back to back on the first frame after A+X / after a reset):
            # only the active solution steps - no candidates, upright tracker, mode logic or output motion. With them
            # the warm-up stalled the teleop loop 2-5 s on the real robot (2026-11-18); the output then moves to the
            # warmed-up solution over the next frames at the VMAX cap.
            lock = {"locked": self.q_home[self.is_waist], "returning": self.waist_ref, "free": None}[self.waist_mode] \
                if self.upright is not None else None
            self._step(self.configuration, 1, lock)     # 1 step per warm-up call: 50 steps in all
            return self.q_out.copy()
        self.frame_n += 1
        tp = np.concatenate([self.tasks[f].transform_target_to_world.translation for f in self.hand_frames
                             if self.tasks[f].transform_target_to_world is not None])
        still = self._tgt_prev is not None and len(tp) == len(self._tgt_prev) and \
            float(np.abs(tp - self._tgt_prev).max()) < 0.001
        self._tgt_prev = tp
        home_w = self.q_home[self.is_waist]
        lock = {"locked": home_w, "returning": self.waist_ref, "free": None}[self.waist_mode]
        # not while the waist straightens: moving the waist reference raises the error for a moment by design
        if self.HOLD_NO_PROGRESS and still and self.waist_mode != "returning":
            q0, e0 = self.configuration.q.copy(), self._task_err(self.configuration)
            self._step(self.configuration, self.num_step_per_frame, lock)
            e1 = self._task_err(self.configuration)
            if e1 > self.HOLD_ERR and e0 - e1 < self.HOLD_MIN_GAIN:
                # target still and out of reach: the step only slides the arm along the reach boundary - keep it
                self.configuration.update(q0)
        else:
            self._step(self.configuration, self.num_step_per_frame, lock)
        if self.upright is not None:
            self._step(self.upright, self.num_step_per_frame, home_w)
            if self.waist_mode == "free" and self.frame_n % 3 == 0:
                # the upright tracker can stall in a bad branch (e.g. hands still): every 3 frames try a fresh start -
                # the active arms with the waist at home, the home pose or a seed - and keep it if it tracks better
                starts = [self.configuration.q.copy(), self.q_home.copy()] + [x.copy() for x in self.seeds[1:]]
                s0 = starts[self._upright_k % len(starts)]
                self._upright_k += 1
                s0[self.is_waist] = home_w
                trial = pink.Configuration(self.upright.model, self.upright.model.createData(), s0)
                self._step(trial, self.RESEED_STEPS, home_w)
                if self._task_err(trial) < self._task_err(self.upright):
                    self.upright = trial
        reseed = self.frame_n % self.RESEED_EVERY == 0
        for k, cfg in enumerate(self.cands):
            if reseed and k == self.reseed_k:
                if self.seed_k < len(self.seeds):
                    s = self.seeds[self.seed_k].copy()
                else:   # random perturbation of the active solution
                    s = np.clip(self.configuration.q + self.armwaist * self.rng.normal(0.0, 0.35, len(self.q_home)),
                                self.lo, self.hi)
                self.seed_k = (self.seed_k + 1) % (len(self.seeds) + 1)
                cfg.update(s)
                self._step(cfg, self.RESEED_STEPS, lock)
                self.better[k] = 0
            else:
                self._step(cfg, self.num_step_per_frame, lock)
        if reseed:
            self.reseed_k = (self.reseed_k + 1) % self.N_CAND

        ta, pa = self._cost(self.configuration)
        if self.upright is not None:
            # mode decisions on the pure hand error (the task cost also counts joints sitting on a limit)
            ta = self._task_err(self.configuration)
            dt_f = 0.1 if self.t_last is None else min(0.2, max(0.02, now - self.t_last))
            if self.waist_mode == "locked":
                # arms alone stay off target: free the waist ("last out")
                self.unlock_n = self.unlock_n + 1 if ta > self.WAIST_UNLOCK_ERR else 0
                if self.unlock_n >= self.WAIST_UNLOCK_FRAMES:
                    self._waist_mode("free", now, ta, f"arms alone stay {ta:.1f} off the target")
            elif self.waist_mode == "free":
                tu = self._task_err(self.upright)
                ok = tu < self.WAIST_RELOCK_ERR and now >= self._retry_after
                self.relock_n = self.relock_n + 1 if ok else 0
                if self.relock_n >= self.WAIST_RELOCK_FRAMES:
                    self.waist_ref = self.configuration.q[self.is_waist].copy()
                    self._waist_mode("returning", now, tu, f"arms reach alone ({tu:.1f}): straightening the waist")
            else:   # returning: ramp the waist reference home while the arms keep the hands on target
                if ta < self.WAIST_RETURN_OK_ERR:
                    step = self.WAIST_BACK_RAD_PER_S * dt_f
                    self.waist_ref = self.waist_ref + np.clip(home_w - self.waist_ref, -step, step)
                self.unlock_n = self.unlock_n + 1 if ta > self.WAIST_UNLOCK_ERR else 0
                if self.unlock_n >= self.WAIST_UNLOCK_FRAMES:
                    self._waist_mode("free", now, ta, f"arms cannot follow the straightening waist ({ta:.1f})")
                    self._retry_after = now + self.WAIST_RETRY_S
                elif float(np.abs(self.waist_ref - home_w).max()) < 1e-4:
                    self._waist_mode("locked", now, ta, "waist upright")
            ta, pa = self._cost(self.configuration)
        best, best_gain, reason = None, 0.0, ""
        for k, cfg in enumerate(self.cands):
            tc, pc = self._cost(cfg)
            if ta - tc >= max(self.TASK_MARGIN, self.TASK_MARGIN_REL * ta):
                why, gain = "tracks better", ta - tc
            elif ta - tc >= -self.TASK_SLACK and pa - pc >= self.POSTURE_MARGIN:
                why, gain = "same tracking, natural posture", (pa - pc) * 0.5
            else:
                self.better[k] = 0
                continue
            self.better[k] += 1
            if self.better[k] >= self.SUSTAIN and gain > best_gain:
                best, best_gain, reason = k, gain, f"{why}: task {ta:.1f}->{tc:.1f} posture {pa:.1f}->{pc:.1f}"
        if best is not None and now >= self.no_switch_until:
            q_from = self.q_out.copy()
            new = self.cands[best]
            self.cands[best] = self.configuration
            self.better[best] = 0
            self.configuration = new
            dq = float(np.max(np.abs(new.q - q_from)))
            self.blend = (now, max(self.BLEND_MIN_S, dq / self.BLEND_RAD_PER_S), q_from)
            self.no_switch_until = now + self.MIN_DWELL_S
            self.switches += 1
            self.last_switch = (now, reason, dq)
            self.switch_log.append((now, dq, self.blend[1], reason))
            if self.verbose:
                print(f"[dwbc_ik] switched IK solution ({reason}), largest joint change {np.degrees(dq):.0f} deg, "
                      f"blended over {self.blend[1]:.1f} s", flush=True)

        q_act = self.configuration.q
        if self.blend is not None:
            t0, dur, q_from = self.blend
            a = min(1.0, (now - t0) / dur)
            a = a * a * (3.0 - 2.0 * a)
            q_want = q_from + a * (q_act - q_from)
            if a >= 1.0:
                self.blend = None
        else:
            q_want = q_act
        dt = 0.1 if self.t_last is None else min(0.2, max(0.02, now - self.t_last))
        self.t_last = now
        if self.OUT_AMAX > 0:
            # bounded-acceleration follower: the speed that can still brake to a stop at the target
            # (sqrt(2 A |e|), capped at VMAX), reached with at most A*dt of speed change per frame, never past the target
            e = q_want - self.q_out
            ae = np.abs(e)
            v_des = np.sign(e) * np.minimum(self.VMAX, np.sqrt(2.0 * self.OUT_AMAX * ae))
            dv = self.OUT_AMAX * dt
            self.v_out = self.v_out + np.clip(v_des - self.v_out, -dv, dv)
            self.v_out = np.clip(self.v_out, -ae / dt, ae / dt)
            self.q_out = self.q_out + self.v_out * dt
        else:
            step = self.VMAX * dt
            self.q_out = self.q_out + np.clip(q_want - self.q_out, -step, step)
        self.robot.cache_forward_kinematics(self.q_out)
        return self.q_out.copy()


# ---------------------------------------------------------------------------------------------------- warm-up patch
def _install_fast_warmup():
    """TeleopRetargetingIK.compute_joint_positions runs the body IK 50x on its warm-up frame; run that warm-up through
    GlobalBodyIKSolver's cheap path (active solution only)."""
    import decoupled_wbc.control.teleop.teleop_retargeting_ik as _TRI

    if getattr(_TRI.TeleopRetargetingIK.compute_joint_positions, "_dwbc_fast", False):
        return
    _orig = _TRI.TeleopRetargetingIK.compute_joint_positions

    def compute_joint_positions(self, body_data, left_hand_data, right_hand_data):
        s = getattr(self, "body_ik_solver", None)
        if self.in_warmup and isinstance(s, GlobalBodyIKSolver):
            s._fast = True
            try:
                return _orig(self, body_data, left_hand_data, right_hand_data)
            finally:
                s._fast = False
        return _orig(self, body_data, left_hand_data, right_hand_data)

    compute_joint_positions._dwbc_fast = True
    _TRI.TeleopRetargetingIK.compute_joint_positions = compute_joint_positions


_install_fast_warmup()
