"""One Isaac process per GPU, independent physical cases within each batch."""
import argparse
import copy
from contextlib import contextmanager, ExitStack
import gzip
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

try:
    from eb.runners import rpo_flat_artifact as flat
    from eb.runners.rpo_bfm_eval import pack_observation, CommandActorSession, sensor_noise_bounds
except ModuleNotFoundError:
    import rpo_flat_artifact as flat
    from rpo_bfm_eval import pack_observation, CommandActorSession, sensor_noise_bounds


def video_integrity(path, quality):
    import imageio.v2 as imageio
    width, height, fps = flat.VIDEO_SETTINGS[quality]
    try:
        with imageio.get_reader(path) as reader:
            metadata = reader.get_meta_data()
            if metadata.get("fps") != fps or tuple(metadata.get("size", ())) != (width, height):
                raise ValueError("BFM video dimensions or frame rate differ")
            frames = 0
            for frame in reader:
                frames += 1
                if frame.shape != (height, width, 3) or frames > 502:
                    raise ValueError("BFM video decoded frames differ from physical horizon")
            if frames < 2:
                raise ValueError("BFM video lacks initial and physical result frames")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("BFM case video is corrupt or incomplete") from exc
    return dict(schema_version="eb/bfm-case-video/v1", sha256=flat.sha256(path),
                decoded_frames=frames, resolution=[width, height], fps=fps)


def partition_seeds(seeds, gpus):
    if gpus not in (1, 4) or len(seeds) < gpus or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Parallel evaluation requires unique nonnegative seeds and 1 or 4 GPUs")
    return [seeds[i::gpus] for i in range(gpus)]


def case_terrain_slot(seed):
    slot = seed % 200
    return slot % 10, slot // 10


@contextmanager
def case_rng(seed, device, state=None):
    import torch
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=devices):
            if state:
                torch.random.set_rng_state(state["cpu"])
                if devices: torch.cuda.set_rng_state(state["cuda"], devices[0])
                np.random.set_state(state["numpy"])
                random.setstate(state["python"])
            else:
                torch.random.default_generator.manual_seed(seed)
                if devices: torch.cuda.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
            try:
                yield
            finally:
                if state is not None:
                    state.update(cpu=torch.random.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
                    if devices: state["cuda"] = torch.cuda.get_rng_state(devices[0])
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def vector_environment_type(partition_invariant=False):
    import torch
    from isaaclab.managers import EventManager
    from rpo_bfm_extension import RPOBFMIsaacEnv

    class EvaluationEnv(RPOBFMIsaacEnv):
        def __init__(self, cfg, case_seeds):
            super().__init__(cfg=cfg, render_mode=None)
            self.case_seeds = case_seeds
            if partition_invariant:
                terrain = self.scene.terrain
                for i, seed in enumerate(case_seeds):
                    level, kind = case_terrain_slot(seed)
                    terrain.terrain_levels[i], terrain.terrain_types[i] = level, kind
                    terrain.env_origins[i] = terrain.terrain_origins[level, kind]
                self.scene.env_origins[:] = terrain.env_origins
            self.command_streams = [{} for _ in case_seeds]
            self.event_streams = [{} for _ in case_seeds]
            self.noise_generators = [torch.Generator(device=self.device).manual_seed(s + 2000003) for s in case_seeds]
            self.noise_keys = [None] * self.num_envs
            self.noise_states = torch.zeros(self.num_envs, 52, device=self.device)
            self.case_managers = []
            self.interval_left = []
            for i, seed in enumerate(case_seeds):
                with case_rng(seed, self.device):
                    manager = EventManager(copy.deepcopy(cfg.events), self)
                    manager.apply("case_startup", env_ids=torch.tensor([i], device=self.device))
                self.case_managers.append(manager)
                intervals = {}
                with case_rng(seed + 3000003, self.device, self.event_streams[i]):
                    for name in manager.active_terms.get("case_interval", []):
                        term = manager.get_term_cfg(name)
                        if term.is_global_time: raise ValueError("Global interval events cannot be independent cases")
                        intervals[name] = random.uniform(*term.interval_range_s)
                self.interval_left.append(intervals)
            original_resample = self.command_generator._resample
            def resample(ids):
                for i in ids:
                    i = int(i)
                    with case_rng(case_seeds[i] + 1000003, self.device, self.command_streams[i]):
                        original_resample(torch.tensor([i], device=self.device))
            self.command_generator._resample = resample
            original_compute = self.command_generator.compute
            def compute(dt):
                original_compute(dt)
                for i, intervals in enumerate(self.interval_left):
                    for name in intervals:
                        intervals[name] -= dt
                        if intervals[name] <= 0:
                            term = self.case_managers[i].get_term_cfg(name)
                            with case_rng(case_seeds[i] + 3000003, self.device, self.event_streams[i]):
                                intervals[name] = random.uniform(*term.interval_range_s)
                                term.func(self, torch.tensor([i], device=self.device), **term.params)
            self.command_generator.compute = compute

        def _reset_idx(self, ids):
            if getattr(self, "_flat_defer_reset", False): return
            if not hasattr(self, "case_seeds"):
                return super()._reset_idx(ids)
            for i in ids:
                i = int(i)
                with case_rng(self.case_seeds[i], self.device):
                    super()._reset_idx(torch.tensor([i], device=self.device))

        def _get_observations(self):
            obs = super()._get_observations()
            if self.cfg.noise.add_noise and hasattr(self, "noise_generators"):
                bounds = torch.from_numpy(sensor_noise_bounds(self.cfg.noise.noise_scales)).to(self.device)
                for i in range(self.num_envs):
                    key = (int(self.bfm_episode_id[i]), int(self.bfm_step_id[i]))
                    if self.noise_keys[i] != key:
                        self.noise_states[i] = obs["states"][i] + (2 * torch.rand(52, device=self.device, generator=self.noise_generators[i]) - 1) * bounds
                        self.noise_keys[i] = key
                obs["states"] = self.noise_states.clone()
            return obs
    return EvaluationEnv


def usd_camera_type():
    from isaaclab.sensors import Camera
    class EvaluationCamera(Camera):
        def _initialize_impl(self):
            super()._initialize_impl()
            # The pinned usdrt Fabric pose view supports only cuda:0; USD keeps rendering on this GPU.
            self._view._use_fabric=False
    return EvaluationCamera


def evaluate_batch(session, env, seeds, output, record_video, quality, fixed_commands=None):
    import torch
    obs, _ = env.reset()
    obs = pack_observation(obs)
    if obs["policy"].shape != (len(seeds), 375) or abs(env.step_dt - flat.CONTROL_PERIOD_S) > 1e-9:
        raise ValueError("BFM batch observation or control period differs")
    from rpo_wbc import RPO_LAB_JOINT_ORDER
    if tuple(env.robot.joint_names) != RPO_LAB_JOINT_ORDER:
        raise ValueError("BFM physical joint order differs")
    limits = flat._hard_limits(env)
    rows = [dict(seed=s, case_id=f"rpo-bfm-seed-{s}", task_id="tsk_rpo_bfm_pretrain", completed=False,
                 success=False, actual_steps=0, episode_len=0, fall=False, nan=False, hard_limit=False,
                 reason="horizon", env_index=i, terrain_seed=env.cfg.scene.terrain.terrain_generator.seed if hasattr(env, "cfg") else seeds[0]) for i, s in enumerate(seeds)]
    active = np.ones(len(seeds), dtype=bool)
    commands, velocities = [[] for _ in seeds], [[] for _ in seeds]
    initial_root = env.robot.data.root_state_w.clone()
    initial_q = env.robot.data.joint_pos.clone()
    initial_dq = torch.zeros_like(env.robot.data.joint_vel)
    fps = flat.VIDEO_SETTINGS[quality][2]
    session.command = lambda: np.where(active[:, None], flat._numpy(env.command_generator.command), 0).astype(np.float32)
    writers, traces = [], []
    (output / "traces").mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if fixed_commands is not None:
            env.command_generator.command[:] = torch.tensor(fixed_commands, device=env.device)
            env.command_generator.is_standing_env[:] = torch.tensor([not any(c) for c in fixed_commands], device=env.device)
        initial_frames = flat._video_frames(env) if record_video else None
        for i, seed in enumerate(seeds):
            initial = flat._initial_state(env, seed, limits, i)
            initial.update(terrain_seed=rows[i]["terrain_seed"], env_index=i, startup_randomization="independent_case_rng")
            path = output / "traces" / f"seed_{seed}.initial.json"
            flat._write_json(path, initial)
            rows[i]["initial_state"] = dict(path=str(path.relative_to(output)), sha256=flat.sha256(path))
            traces.append(stack.enter_context(gzip.open(output / "traces" / f"seed_{seed}.jsonl.gz", "wt")))
            writer = None
            if record_video:
                import imageio.v2 as imageio
                video_rel = f"seed_{seed}/0.mp4"
                path = output / "video/eval" / video_rel
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1,
                    **({"quality": None, "ffmpeg_params": ["-crf", "18", "-preset", "fast"]} if quality == "hd" else {}))
                stack.callback(writer.close)
                writer.append_data(initial_frames[i])
                rows[i]["video_rel"] = video_rel
            writers.append(writer)
            if fixed_commands is not None: rows[i]["diagnostic_command"] = fixed_commands[i]
        for step in range(flat.REQUIRED_STEPS):
            if not active.any(): break
            if fixed_commands is not None: env.command_generator.command[:] = torch.tensor(fixed_commands, device=env.device)
            pre_commands = flat._numpy(env.command_generator.command).copy()
            headings = flat._numpy(env.command_generator.heading_target).copy()
            standing = flat._numpy(env.command_generator.is_standing_env).copy()
            for i in np.flatnonzero(active):
                if not (np.isfinite(obs["policy"][i]).all() and np.isfinite(obs["critic"][i]).all()
                        and np.isfinite(pre_commands[i]).all() and np.isfinite(headings[i])):
                    rows[i].update(nan=True, reason="non_finite_policy_input")
                    active[i] = False
                    traces[i].write(json.dumps(dict(step=step, advanced=False, nan=True)) + "\n")
            if not active.any(): break
            actor_obs = np.where(active[:, None], obs["policy"], 0).astype(np.float32)
            action = session.run(None, {session.get_inputs()[0].name: actor_obs})[0].astype(np.float32)
            if action.shape != (len(seeds), 23): raise ValueError("BFM batched actor shape differs")
            for i in np.flatnonzero(active):
                if not np.isfinite(action[i]).all() or np.any(np.abs(action[i]) > 1.000001):
                    rows[i].update(nan=not np.isfinite(action[i]).all(), hard_limit=bool(np.isfinite(action[i]).all()), reason="invalid_policy_output")
                    active[i] = False
                    traces[i].write(json.dumps(dict(step=step, advanced=False, nan=rows[i]["nan"], hard_limit=rows[i]["hard_limit"])) + "\n")
            if not active.any(): break
            inactive = torch.tensor(np.flatnonzero(~active), device=env.device, dtype=torch.long)
            if len(inactive):
                roots = initial_root[inactive].clone()
                roots[:, 7:] = 0
                env.robot.write_root_state_to_sim(roots, env_ids=inactive)
                env.robot.write_joint_state_to_sim(initial_q[inactive], initial_dq[inactive], env_ids=inactive)
            action[~active] = 0
            obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
            obs = pack_observation(obs)
            terminated, truncated = flat._numpy(terminated), flat._numpy(truncated)
            frame_indices = []
            for i in np.flatnonzero(active):
                row = rows[i]
                row["actual_steps"] += 1
                row["episode_len"] = row["actual_steps"]
                state = flat._state(env, i)
                row["nan"] = not (all(np.isfinite(v).all() for v in state.values()) and np.isfinite(obs["policy"][i]).all() and np.isfinite(obs["critic"][i]).all())
                row["hard_limit"] = flat._violates_hard_limit(state, limits)
                velocity = flat.command_frame_velocity(state)
                commands[i].append(pre_commands[i])
                velocities[i].append(velocity)
                row["fall"] = bool(terminated[i] or state["root_pos_w"][2] - float(env.scene.env_origins[i, 2]) < .5 or -state["projected_gravity_b"][2] < math.cos(.8))
                done = bool(terminated[i] or truncated[i])
                stop = row["nan"] or row["hard_limit"] or row["fall"] or done or step + 1 == flat.REQUIRED_STEPS
                entry = dict(step=step + 1, time_s=(step + 1) * flat.CONTROL_PERIOD_S, advanced=True,
                    command_pre=flat._trace_array(pre_commands[i]), command_post=flat._trace_array(flat._numpy(env.command_generator.command[i])),
                    metric_velocity=flat._trace_array(velocity), heading_target_pre=float(headings[i]), standing_pre=bool(standing[i]),
                    action=flat._trace_array(action[i]), z=flat._trace_array(session.last_latent[i]),
                    terminated=bool(terminated[i]), truncated=bool(truncated[i]), nan=row["nan"], fall=row["fall"], hard_limit=row["hard_limit"],
                    **{k: flat._trace_array(v) for k, v in state.items()})
                if getattr(env, "bfm_slew", False):
                    entry["applied_action"] = flat._trace_array(flat._numpy(env.bfm_last_action)[i])
                traces[i].write(json.dumps(entry, allow_nan=False) + "\n")
                if writers[i] is not None and (stop or (step + 1) % round(1 / (flat.CONTROL_PERIOD_S * fps)) == 0): frame_indices.append(i)
                if stop:
                    row["reason"] = "non_finite_state" if row["nan"] else "physical_hard_limit" if row["hard_limit"] else "fall" if row["fall"] else "horizon" if step + 1 == flat.REQUIRED_STEPS else "early_timeout"
                    active[i] = False
            if frame_indices:
                frames = flat._video_frames(env)
                for i in frame_indices: writers[i].append_data(frames[i])
    for i, row in enumerate(rows):
        row.update(completed=True, **flat.tracking_metrics(commands[i], velocities[i]))
        row["success"] = flat.case_success(row)
        row["status"] = "success" if row["success"] else "task_failure"
        if row["reason"] == "horizon" and not row["success"]: row["reason"] = "velocity_tracking_error"
        row["termination_reason"] = row["reason"]
        path = output / "traces" / f"seed_{row['seed']}.jsonl.gz"
        row["trace"] = dict(path=str(path.relative_to(output)), sha256=flat.sha256(path), policy_steps=row["actual_steps"], metric_steps=row["metric_steps"])
        if record_video:
            row["video_integrity"] = video_integrity(output / "video/eval" / row["video_rel"], quality)
            stride = round(1 / (flat.CONTROL_PERIOD_S * fps))
            if row["video_integrity"]["decoded_frames"] != 1 + math.ceil(row["actual_steps"] / stride):
                raise ValueError("BFM video frames do not cover the actual physical case")
    return rows


def worker(args, seeds):
    import onnxruntime as ort
    import torch
    from isaaclab.app import AppLauncher
    device=f"cuda:{args.worker_rank}"
    torch.cuda.set_device(args.worker_rank)
    app = AppLauncher(headless=True, enable_cameras=args.record_video, device=device, multi_gpu=False).app
    import omni.usd
    from isaaclab.sensors import RayCaster
    from rpo_bfm_extension import RPOBFMEnvCfg
    from rpo_bfm_eval import configure_action_control, standard_configuration
    configure_action_control(json.loads((args.output / "exported/export-manifest.json").read_text()))
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    session = CommandActorSession(ort.InferenceSession(str(args.output / "exported/policy.onnx"), sess_options=options, providers=["CPUExecutionProvider"]),
        ort.InferenceSession(str(args.output / "exported/command-to-latent.onnx"), sess_options=options, providers=["CPUExecutionProvider"]))
    root = args.output / "workers" / f"rank_{args.worker_rank}"
    root.mkdir(parents=True, exist_ok=True)
    rows, batches = [], []
    for start in range(0, len(seeds), args.envs_per_gpu):
        batch = seeds[start:start + args.envs_per_gpu]
        RayCaster.meshes.clear()
        omni.usd.get_context().new_stage()
        cfg = RPOBFMEnvCfg()
        if not args.diagnostic_commands: cfg = standard_configuration(cfg)
        # Fixed scene seeds also fix the converter's random temporary directory suffix.
        cfg.scene.robot.spawn.usd_dir = str(root / "robot-usd")
        cfg.scene.num_envs = len(batch)
        cfg.sim.device=device
        cfg.scene.env_spacing = 10
        terrain_seed = 64000 if args.partition_invariant else batch[0]
        cfg.seed = terrain_seed
        cfg.scene.terrain.terrain_generator.seed = terrain_seed
        cfg.events = copy.deepcopy(cfg.events)
        for term in vars(cfg.events).values():
            if getattr(term, "mode", None) == "startup": term.mode = "case_startup"
            elif getattr(term, "mode", None) == "interval": term.mode = "case_interval"
        if args.record_video:
            flat._add_camera(cfg, args.video_quality)
            cfg.scene.eval_camera.prim_path="{ENV_REGEX_NS}/EvalCamera"
            cfg.scene.eval_camera.class_type=usd_camera_type()
        settings = dict(commands=cfg.commands.to_dict(), noise=cfg.noise.to_dict(), events=cfg.events.to_dict(),
            terrain=cfg.scene.terrain.terrain_generator.to_dict(), startup_randomization="independent_case_rng",
            control_period_s=cfg.decimation * cfg.sim.dt, episode_length_s=cfg.episode_length_s)
        settings = json.loads(json.dumps(settings, default=lambda v: f"{v.__module__}.{v.__qualname__}" if callable(v) else str(v)))
        env = vector_environment_type(args.partition_invariant)(cfg, batch)
        try:
            fixed = [([0., 0., 0.], [.3, 0., 0.], [0., .2, 0.], [.2, 0., .5])[s - 60000] for s in batch] if args.diagnostic_commands else None
            batch_rows = evaluate_batch(session, env, batch, args.output, args.record_video, args.video_quality, fixed)
            batches.append(dict(seeds=batch, terrain_seed=terrain_seed, num_envs=len(batch), environment_settings=settings))
            for row in batch_rows:
                row.update(worker_rank=args.worker_rank, run_id=os.environ.get("EB_RUN_ID"), plan_hash=args.plan_hash)
                with (root / "eval-cases.jsonl").open("a") as stream: stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(dict(case=row["case_id"], status=row["status"], actual_steps=row["actual_steps"])), flush=True)
                rows.append(row)
        finally:
            env.close()
    flat._write_json(root / "worker-manifest.json", dict(worker_rank=args.worker_rank, pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_device_index=args.worker_rank,
        gpu_uuid=str(torch.cuda.get_device_properties(args.worker_rank).uuid), gpu_name=torch.cuda.get_device_name(args.worker_rank),
        seeds=seeds, batches=batches, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), **flat.summarize_cases(rows, len(seeds))))
    app.close()


def main():
    p = argparse.ArgumentParser()
    for key in ("checkpoint", "resolved-config", "motion-root"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--output", type=Path, required=True)
    for key in ("asset-sha256", "weights-sha256", "motion-sha256", "config-sha256"):
        p.add_argument("--" + key)
    p.add_argument("--plan-hash", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--eval-gpus", type=int, choices=(1, 4), default=1)
    p.add_argument("--envs-per-gpu", type=int, choices=(1, 4), default=4)
    p.add_argument("--worker-rank", type=int)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-quality", choices=flat.VIDEO_SETTINGS, default="standard")
    p.add_argument("--diagnostic-commands", action="store_true")
    p.add_argument("--partition-invariant", action="store_true")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.worker_rank is not None:
        worker(a, seeds)
        return
    groups = partition_seeds(seeds, a.eval_gpus)
    if (a.output / "eval-cases.jsonl").exists(): raise ValueError("Evaluation output already exists")
    from rpo_bfm_artifact import export
    native = (a.checkpoint / "native-checkpoint.json").is_file()
    if native:
        from bfm_ufo_artifact import native_checkpoint_path, export_ufo_policy
        if not a.partition_invariant and (a.eval_gpus != 1 or a.envs_per_gpu != 1):
            raise ValueError("UFO parallel evaluation topology has not passed qualification")
        weights = native_checkpoint_path(a.checkpoint) / "model/model.safetensors"
    else:
        weights = a.checkpoint / "model_state_dict/full_weights.pt"
    if (flat.sha256(weights) != a.weights_sha256
            or flat.sha256(a.resolved_config) != a.config_sha256 or flat.sha256(a.motion_root / "manifest.json") != a.motion_sha256):
        raise ValueError("Parallel evaluation frozen inputs changed")
    (export_ufo_policy if native else export)(a.checkpoint, a.resolved_config, a.motion_root, a.asset_sha256, a.output / "exported")
    exported = json.loads((a.output / "exported/export-manifest.json").read_text())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible[0]: visible = [str(i) for i in range(a.eval_gpus)]
    if len(visible) < a.eval_gpus: raise ValueError("Assigned CUDA devices fewer than frozen GPU count")
    processes = []
    with ExitStack() as stack:
        for rank, group in enumerate(groups):
            root = a.output / "workers" / f"rank_{rank}"
            root.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__)), "--output", str(a.output), "--seeds", ",".join(map(str, group)),
                "--plan-hash", a.plan_hash, "--worker-rank", str(rank), "--envs-per-gpu", str(a.envs_per_gpu), "--video-quality", a.video_quality]
            if a.record_video: command.append("--record-video")
            if a.diagnostic_commands: command.append("--diagnostic-commands")
            if a.partition_invariant: command.append("--partition-invariant")
            log = stack.enter_context((root / "console.log").open("w"))
            processes.append(subprocess.Popen(command, env={**os.environ,
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PXR_WORK_THREAD_LIMIT":"16"}, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    if any(codes): raise RuntimeError(f"Parallel evaluation workers failed: {codes}; inspect workers/rank_*/console.log")
    rows, workers = [], []
    for rank, group in enumerate(groups):
        root = a.output / "workers" / f"rank_{rank}"
        receipt = json.loads((root / "worker-manifest.json").read_text())
        if receipt["seeds"] != group: raise ValueError("Worker seed shard changed")
        workers.append(receipt)
        rows.extend(json.loads(line) for line in (root / "eval-cases.jsonl").read_text().splitlines())
    if sorted(row["seed"] for row in rows) != sorted(seeds): raise ValueError("Parallel cases missing or duplicated")
    rows.sort(key=lambda row: seeds.index(row["seed"]))
    (a.output / "eval-cases.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    receipt = dict(schema_version="eb/rpo-bfm-eval/v1", backend="isaac", criterion="velocity_commands", required_steps=flat.REQUIRED_STEPS,
        plan_hash=a.plan_hash, run_id=os.environ.get("EB_RUN_ID"), record_video=a.record_video, video_quality=a.video_quality,
        video_settings=dict(zip(("width", "height", "fps"), flat.VIDEO_SETTINGS[a.video_quality])) if a.record_video else None,
        seeds=seeds, official_revision=flat.ROBOLAB_REVISION, policy="deterministic_onnx", velocity_frame="yaw_aligned", yaw_rate_frame="world_z",
        onnx=exported["onnx"], command_latent=exported["command_latent"], codec_sha256=exported["codec_sha256"],
        action_control=exported.get("action_control"),
        **({"auxiliary_profile": exported["auxiliary_profile"]} if exported.get("auxiliary_profile") else {}),
        diagnostic_commands=a.diagnostic_commands, asset_sha256=exported["asset_sha256"], motion_manifest_sha256=exported["motion_manifest_sha256"],
        parallelism=dict(gpus=a.eval_gpus, envs_per_gpu=a.envs_per_gpu, protocol="independent_cases_fixed_terrain_v2" if a.partition_invariant else "independent_cases_shared_terrain_v1"),
        workers=workers, environment_settings=[w["batches"] for w in workers],
        code_trace=[dict(path=str(path), sha256=flat.sha256(path)) for path in
                    (Path(__file__), Path(flat.__file__), Path(__file__).parent / "rpo_bfm_eval.py", Path(__file__).parent / "rpo_bfm_extension.py")],
        **flat.summarize_cases(rows, len(seeds)))
    flat._write_json(a.output / "eval-manifest.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__": main()
