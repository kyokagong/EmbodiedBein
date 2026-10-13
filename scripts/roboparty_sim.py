#!/usr/bin/env python3
"""Standalone RPO-Flat ONNX keyboard preview in MuJoCo (no platform imports)."""

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import onnxruntime as ort


JOINT_ORDER = (
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint",
    "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint",
    "left_arm_roll_joint", "right_arm_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_arm_yaw_joint", "right_arm_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)
PERIOD = 0.02
COMMAND_LOW = np.array([-0.6, -0.5, -1.57])
COMMAND_HIGH = np.array([1.0, 0.5, 1.57])
HELP = "W/S: vx +/-0.1 | A/D: vy +/-0.1 | Q/E: yaw +/-0.2 | X: stop | P: pause | R: reset | Esc: exit"


def joint_parameters(name):
    """Nominal RoboLab RPO_CFG: default angle, stiffness, damping, effort limit."""
    if name == "torso_joint" or "knee" in name:
        return (0.3 if "knee" in name else 0.0), 150.0, 5.0, 120.0
    if "thigh" in name:
        return (-0.1 if "pitch" in name else 0.0), 100.0, 3.3, 120.0
    if "ankle" in name:
        return (-0.2 if "pitch" in name else 0.0), 40.0, 2.0, 27.0
    if "elbow_pitch" in name:
        return 0.78, 30.0, 1.5, 27.0
    if "elbow_yaw" in name:
        return 0.0, 20.0, 1.0, 27.0
    if "arm" in name:
        angle = 0.18 if "pitch" in name else 0.0
        if "roll" in name:
            angle = 0.06 if name.startswith("left_") else -0.06
        return angle, 40.0, 2.0, 27.0
    raise ValueError(f"Unsupported joint: {name}")


def load_policy(path):
    manifest = json.loads(path.with_name("export-manifest.json").read_text())
    codec = manifest["codec"]
    if (manifest.get("status") != "verified" or codec["joint_order"] != list(JOINT_ORDER)
            or codec["normalization"] != "embedded" or codec["history_length"] != 10
            or codec["frame_dim"] != 78 or codec["action_scale"] != 0.25):
        raise ValueError("Requires an RPO-Flat 780x23 export with embedded normalization and native joint order")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["onnx"]["sha256"]:
        raise ValueError("ONNX hash differs from export-manifest.json")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Policy must have one input and one output")
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if (inp.name, inp.type, len(inp.shape), inp.shape[-1]) != ("observation", "tensor(float)", 2, 780):
        raise ValueError("Policy input must be observation: float32[batch,780]")
    if (out.name, out.type, len(out.shape), out.shape[-1]) != ("action", "tensor(float)", 2, 23):
        raise ValueError("Policy output must be action: float32[batch,23]")
    golden_path = path.with_name("golden-traces.npz")
    if hashlib.sha256(golden_path.read_bytes()).hexdigest() != manifest["golden"]["sha256"]:
        raise ValueError("Golden trace hash differs from export-manifest.json")
    with np.load(golden_path, allow_pickle=False) as golden:
        actual = np.concatenate([session.run(["action"], {"observation": row[None]})[0]
                                 for row in golden["observation"]])
        expected = golden["torch_action_single"]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
        if not np.isfinite(actual).all() or .25 * np.max(np.abs(
                np.clip(actual, -100, 100) - np.clip(expected, -100, 100))) > 1e-4:
            raise ValueError("ONNX failed finite/joint-target golden check")
    print(f"ONNX verified: {path.name}, CPUExecutionProvider, 780 -> 23", flush=True)
    return session


class Keyboard:
    def __init__(self, command):
        self.command = np.array(command, dtype=np.float32)
        self.paused = self.reset_requested = False
        self.quit_requested = False
        self.lock = threading.Lock()

    def on_key(self, key):
        with self.lock:
            directions = {ord("W"): (0, .1), ord("S"): (0, -.1), ord("A"): (1, .1),
                          ord("D"): (1, -.1), ord("Q"): (2, .2), ord("E"): (2, -.2)}
            if key in directions:
                axis, change = directions[key]
                self.command[axis] = np.clip(self.command[axis] + change, COMMAND_LOW[axis], COMMAND_HIGH[axis])
            elif key == ord("X"):
                self.command.fill(0)
            elif key == ord("P"):
                self.paused = not self.paused
            elif key == ord("R"):
                self.command.fill(0)
                self.reset_requested, self.paused = True, False
            elif key == 256:  # GLFW_KEY_ESCAPE
                self.quit_requested = True
            else:
                return
            print(f"KEY: vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} wz={self.command[2]:+.2f} paused={self.paused}", flush=True)


class Simulation:
    def __init__(self, session, mjcf, dt, delay_ms):
        self.session = session
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        m = self.model
        self.decimation = round(PERIOD / dt)
        m.opt.timestep = dt
        self.delay_steps = round(delay_ms / 1000 / dt)
        self.data = mujoco.MjData(m)
        self.base = m.body("base_link").id
        joints = np.array([m.joint(name).id for name in JOINT_ORDER])
        self.actuators = np.array([m.actuator(name).id for name in JOINT_ORDER])
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if (len(free) != 1 or m.nu != 23 or m.njnt != 24 or m.jnt_bodyid[free[0]] != self.base
                or np.any(m.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)
                or np.any(m.actuator_trntype[self.actuators] != mujoco.mjtTrn.mjTRN_JOINT)
                or not np.array_equal(m.actuator_trnid[self.actuators, 0], joints)
                or not np.allclose(m.actuator_gear[self.actuators], [1, 0, 0, 0, 0, 0])
                or np.any(m.actuator_dyntype[self.actuators] != mujoco.mjtDyn.mjDYN_NONE)
                or np.any(m.actuator_gaintype[self.actuators] != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(m.actuator_biastype[self.actuators] != mujoco.mjtBias.mjBIAS_NONE)
                or not np.allclose(m.actuator_gainprm[self.actuators, 0], 1)):
            raise ValueError("MJCF must have one free base and 23 named hinge joints with unit-gear torque motors")
        self.root_qpos = int(m.jnt_qposadr[free[0]])
        self.root_dof = int(m.jnt_dofadr[free[0]])
        self.qpos = m.jnt_qposadr[joints]
        self.qvel = m.jnt_dofadr[joints]
        self.default, self.kp, self.kd, self.effort = np.array([joint_parameters(n) for n in JOINT_ORDER]).T
        m.actuator_ctrllimited[self.actuators] = 1
        m.actuator_ctrlrange[self.actuators] = np.column_stack((-self.effort, self.effort))
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = self.default
        self.data.qpos[self.root_qpos:self.root_qpos + 7] = [0, 0, .75, 1, 0, 0, 0]
        self.history = None
        self.last_action = np.zeros(23, dtype=np.float32)
        self.targets = deque([self.default.copy() for _ in range(self.delay_steps + 1)],
                             maxlen=self.delay_steps + 1)
        mujoco.mj_forward(self.model, self.data)

    def frame(self, command):
        velocity = np.empty(6)
        # BODY's local velocity uses the inertial frame; explicitly rotate world -> base link.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        world_to_base = self.data.xmat[self.base].reshape(3, 3).T
        gravity = world_to_base @ np.array([0., 0., -1.])
        return np.concatenate((world_to_base @ velocity[:3], gravity, command, self.data.qpos[self.qpos] - self.default,
                               self.data.qvel[self.qvel], self.last_action)).astype(np.float32)

    def step(self, command):
        mujoco.mj_forward(self.model, self.data)
        frame = self.frame(command)
        if not np.isfinite(frame).all():
            raise ValueError("Non-finite observation")
        if self.history is None:
            self.history = np.tile(frame, (10, 1))
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        obs = np.clip(self.history.reshape(1, 780), -100, 100)
        action = self.session.run(["action"], {"observation": obs})[0]
        if action.shape != (1, 23) or not np.isfinite(action).all():
            raise ValueError("Non-finite or incorrectly shaped policy action")
        self.last_action = action[0].copy()  # History contains raw actions, before actuator clipping.
        target = self.default + .25 * np.clip(self.last_action, -100, 100)
        for _ in range(self.decimation):
            self.targets.append(target.copy())
            torque = self.kp * (self.targets[0] - self.data.qpos[self.qpos]) - self.kd * self.data.qvel[self.qvel]
            self.data.ctrl[self.actuators] = np.clip(torque, -self.effort, self.effort)
            mujoco.mj_step(self.model, self.data)
        bad_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(self.data.warning[w].number for w in bad_warnings)):
            raise ValueError("MuJoCo reported an unstable/non-finite physics state")
        mujoco.mj_forward(self.model, self.data)

    def fallen(self):
        return bool(self.data.xpos[self.base, 2] < .50 or self.data.xmat[self.base].reshape(3, 3)[2, 2] < math.cos(.8))


def self_test(sim):
    """One runnable check of keyboard, frame convention, mapping, history and reset."""
    keys = Keyboard([0, 0, 0])
    for _ in range(11):
        keys.on_key(ord("W"))
    np.testing.assert_allclose(keys.command, [1, 0, 0])
    keys.on_key(ord("A"))
    keys.on_key(ord("Q"))
    np.testing.assert_allclose(keys.command, [1, .1, .2])
    keys.on_key(ord("X"))
    assert not keys.command.any()
    keys.on_key(ord("P"))
    assert keys.paused
    keys.on_key(ord("R"))
    assert keys.reset_requested and not keys.paused
    keys.on_key(256)
    assert keys.quit_requested
    sim.reset()
    sim.data.qpos[sim.root_qpos + 3:sim.root_qpos + 7] = [math.cos(math.pi / 12), math.sin(math.pi / 12), 0, 0]
    sim.data.qvel[sim.root_dof + 3:sim.root_dof + 6] = [.1, .2, .3]
    offsets = np.arange(23) * .001
    sim.data.qpos[sim.qpos] += offsets
    sim.data.qvel[sim.qvel] = np.arange(23) * .01
    mujoco.mj_forward(sim.model, sim.data)
    frame = sim.frame([.2, -.1, .3])
    np.testing.assert_allclose(frame[:9], [.1, .2, .3, 0, -.5, -math.sqrt(.75), .2, -.1, .3], atol=1e-6)
    np.testing.assert_allclose(frame[9:32], offsets, atol=1e-6)
    np.testing.assert_allclose(frame[32:55], np.arange(23) * .01, atol=1e-6)
    sim.reset()
    first = sim.frame([0, 0, 0])
    sim.step([0, 0, 0])
    np.testing.assert_allclose(sim.history, np.tile(first, (10, 1)))
    raw = sim.last_action.copy()
    sim.step([.1, 0, 0])
    np.testing.assert_allclose(sim.history[-1, 55:78], raw)
    np.testing.assert_allclose(sim.history[:-1], np.tile(first, (9, 1)))
    sim.reset()
    assert sim.history is None and not sim.last_action.any() and sim.data.time == 0
    sim.step([0, 0, 0])
    np.testing.assert_allclose(sim.history, np.tile(first, (10, 1)))
    print("SELF-TEST PASS: keyboard, body-frame observation, named joints, golden ONNX, history, reset", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("/models/policy.onnx"))
    parser.add_argument("--mjcf", type=Path, default=Path("/robot/mjcf/rpo.xml"))
    parser.add_argument("--headless", action="store_true", help="Run without a window; defaults to 10 simulation seconds")
    parser.add_argument("--seconds", type=float, help="Stop after this many simulation seconds")
    parser.add_argument("--command", nargs=3, type=float, default=[0, 0, 0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--physics-dt", type=float, default=.001, help="MuJoCo timestep; must divide 0.02 seconds")
    parser.add_argument("--actuator-delay-ms", type=float, default=0, help="Fixed PD target delay, rounded to physics steps")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.physics_dt) or not .0001 <= args.physics_dt <= PERIOD
            or not math.isclose(round(PERIOD / args.physics_dt) * args.physics_dt, PERIOD, abs_tol=1e-9)
            or not math.isfinite(args.actuator_delay_ms) or not 0 <= args.actuator_delay_ms <= 100
            or args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0)
            or not np.isfinite(args.command).all()
            or np.any(np.array(args.command) < COMMAND_LOW) or np.any(np.array(args.command) > COMMAND_HIGH)):
        parser.error("Invalid timestep/delay/duration/command; command bounds: vx[-.6,1], vy[-.5,.5], wz[-1.57,1.57]")
    if not args.headless and not args.self_test and not os.environ.get("DISPLAY"):
        parser.error("A desktop DISPLAY is required; use --headless for an SSH-only physics check")
    sim = Simulation(load_policy(args.policy), args.mjcf, args.physics_dt, args.actuator_delay_ms)
    if args.self_test:
        self_test(sim)
        return 0
    if args.headless and args.seconds is None:
        args.seconds = 10
    keys = Keyboard(args.command)
    print(HELP, flush=True)
    print(f"Policy 50 Hz | physics {1 / args.physics_dt:g} Hz | delay {sim.delay_steps * args.physics_dt * 1000:g} ms", flush=True)
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.on_key)
    elapsed, fallen, last_status = 0.0, False, -1.0
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3, 135, -20
        while viewer is None or viewer.is_running():
            start = time.monotonic()
            with keys.lock:
                command, paused, reset = keys.command.copy(), keys.paused, keys.reset_requested
                quit_requested = keys.quit_requested
                keys.reset_requested = False
            if quit_requested:
                break
            with viewer.lock() if viewer is not None else nullcontext():
                if reset:
                    sim.reset()
                    fallen = False
                    print("RESET: pose, velocity, raw action and observation history cleared", flush=True)
                if not paused:
                    sim.step(command)
                    elapsed += PERIOD
                    fallen = sim.fallen()
                    if fallen:
                        with keys.lock:
                            keys.paused = True
                            keys.command.fill(0)
                        print("FALL: simulation paused; press R to reset", flush=True)
                if viewer is not None:
                    viewer.cam.lookat[:] = sim.data.xpos[sim.base]
            if viewer is not None:
                viewer.sync()
            if elapsed - last_status >= 1:
                position = sim.data.xpos[sim.base]
                print(f"t={elapsed:.2f}s vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f} x={position[0]:+.2f}m y={position[1]:+.2f}m height={position[2]:.3f}m paused={paused}", flush=True)
                last_status = elapsed
            if (viewer is None and fallen) or (args.seconds is not None and elapsed >= args.seconds - 1e-9):
                break
            if viewer is not None:
                time.sleep(max(0, PERIOD - (time.monotonic() - start)))
    print(json.dumps({"simulated_seconds": round(elapsed, 3), "fallen": fallen, "finite": True,
                      "base_position_m": [round(float(v), 3) for v in sim.data.xpos[sim.base]]}), flush=True)
    return int(fallen)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"Simulation failed: {error}") from error
