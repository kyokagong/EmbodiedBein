"""RLinf RPC lifecycle; pinned UFO is the sole model/optimizer owner."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from rlinf.data.schema.embodied_types import convert_trajectories_to_batch
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement

try:
    from eb.runners.bfm_ufo_agent import (
        AGENT_CONFIG_KEYS,
        AUX_NAMES,
        OBS_DIMS,
        VELOCITY_AUX_NAMES,
        build_agent,
        instrument_updates,
        optimizer_steps,
    )
    from eb.runners.bfm_ufo_policy import (
        observation_space,
        pool_state,
        sync_identity,
        validate_sampler_identities,
    )
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.rpo_ufo_motion import load_robot_state
except ModuleNotFoundError:
    from bfm_ufo_agent import (
        AGENT_CONFIG_KEYS,
        AUX_NAMES,
        OBS_DIMS,
        VELOCITY_AUX_NAMES,
        build_agent,
        instrument_updates,
        optimizer_steps,
    )
    from bfm_ufo_policy import observation_space, pool_state, sync_identity, validate_sampler_identities
    from bfm_ufo_replay import UFOReplayBridge
    from rpo_ufo_motion import load_robot_state


def expert_codec_hash(model_cfg):
    try:
        from eb.rpo_bfm import CODEC_HASH, CODEC_V2_HASH
    except ModuleNotFoundError:
        from rpo_bfm import CODEC_HASH, CODEC_V2_HASH
    return CODEC_V2_HASH if model_cfg.get("control_profile") == "reachable_v2" else CODEC_HASH


def validate_ufo_runtime_config(cfg):
    model = cfg.actor.model
    native_profile = model.get("physics_profile")
    mjlab = native_profile in {"native_mjcf_v12", "native_mjcf_viability_v13"}
    tracking = model.get("context_profile") == "environment_tracking_v10"
    ground = model.get("ground_profile") == "wide_plane_v9"
    contact = model.get("contact_profile") == "contact_stable_v8"
    slew = model.get("learning_profile") == "physical_limits_stable_slew_v4"
    reachable = model.get("optimizer_profile") == "stabilized_v7"
    velocity = model.get("auxiliary_profile") in {"soft_velocity_v5", "soft_velocity_v7"}
    scale = model.get("scaling_profile") == "env6144_batch2048_v6"
    stable = slew or model.get("learning_profile") == "physical_limits_stable_latent_v3"
    safe = stable or model.get("learning_profile") == "physical_limits_v2"
    if (
        cfg.actor.training_backend != "ufo_agent"
        or cfg.algorithm.loss_type != "ufo_fb_cpr_aux"
        or cfg.actor.optim is not None
        or cfg.actor.fsdp_config is not None
        or model.hidden_dim != (2048 if stable else 1024)
        or model.hidden_layers != (6 if stable else 3)
        or model.batch_size != (2048 if scale else 1024 if stable else 128)
        or model.get("scaling_profile") != ("env6144_batch2048_v6" if scale and slew and velocity else None)
        or model.get("architecture_profile") != ("native_fb_v1" if stable else None)
        or model.get("rollout_expert_trajectories", True) is not (tracking or not stable)
        or model.get("physics_profile") != (native_profile if mjlab and tracking else None)
        or cfg.env.train.env_type != ("mjlab" if mjlab else "isaaclab")
        or model.get("context_profile") != ("environment_tracking_v10" if tracking and (ground or mjlab) else None)
        or model.get("auxiliary_profile") != ("soft_velocity_v7" if reachable else "soft_velocity_v5" if velocity and slew else None)
        or model.get("optimizer_profile") != ("stabilized_v7" if reachable and scale and slew and velocity else None)
        or model.get("contact_profile") != (native_profile if mjlab and reachable else "contact_stable_v8" if contact and reachable else None)
        or model.get("ground_profile") != ("wide_plane_v9" if ground and contact else None)
        or model.get("control_profile") != ("reachable_v2" if reachable else None)
        or (reachable and model.get("codec_sha256") != expert_codec_hash(model))
        or cfg.actor.global_batch_size != model.batch_size
        or cfg.actor.micro_batch_size != model.batch_size
        or model.replay_capacity != (4000000 if tracking else 1000000 if stable else 100000)
        or cfg.runner.use_training_pipeline
        or cfg.runner.max_epochs != 4096
        or cfg.runner.max_steps != -1
        or cfg.env.train.total_num_envs
        != (6144 if scale and model.get("topology_id") == "one_learner_three_samplers" else 192 if model.get("topology_id") == "one_learner_three_samplers" else 16)
        or cfg.env.train.max_steps_per_rollout_epoch != 4
        or cfg.env.train.init_params.id
        != (
            ("RPO-UFO-MJLab-RSI-viability-v13" if native_profile == "native_mjcf_viability_v13" else "RPO-UFO-MJLab-RSI-v12") if mjlab else
            "RPO-UFO-Isaac-RSI-tracking-contact-v10" if tracking else
            "RPO-UFO-Isaac-RSI-ground-v9" if ground
            else "RPO-UFO-Isaac-RSI-contact-v8"
            if contact
            else "RPO-UFO-Isaac-RSI-reachable-v7"
            if reachable
            else "RPO-UFO-Isaac-RSI-slew-velocity-v5"
            if velocity
            else "RPO-UFO-Isaac-RSI-slew-v4"
            if slew
            else "RPO-UFO-Isaac-RSI-safe-v2"
            if safe
            else "RPO-UFO-Isaac-RSI-v1"
        )
    ):
        raise ValueError("UFO resolved runtime differs from approved native recipe")


class UFOLearner:
    def __init__(self, agent, replay, expert, *, target, sample_interval=1024, optimizer_profile=None):
        if type(target) is not int or target < 0:
            raise ValueError("UFO requires a nonnegative absolute update target")
        self.agent, self.replay, self.expert = agent, replay, expert
        if sample_interval not in (1024, 24576):
            raise ValueError("Unapproved UFO sampling interval")
        self.sample_interval = sample_interval
        self.target, self.joint_updates, self.sampling_rounds, self.next_trigger = target, 0, 0, 10240 + sample_interval
        self.started = time.monotonic()
        self.writer = None
        self.diagnostics = instrument_updates(agent, optimizer_profile=optimizer_profile) if hasattr(agent, "_model") else {}

    def train_round(self, transitions):
        collected = self.replay.append(transitions)
        self.sampling_rounds += 1
        metrics = {}
        while self.replay.received >= self.next_trigger and self.joint_updates < self.target:
            for _ in range(min(16, self.target - self.joint_updates)):
                native = self.agent.update(
                    {"train": self.replay, "expert_slicer": self.expert}, self.replay.received
                )
                if any(not torch.isfinite(value).all() for value in native.values()):
                    raise ValueError("Nonfinite UFO update metric")
                self.joint_updates += 1
                if set(optimizer_steps(self.agent).values()) != {self.joint_updates}:
                    raise ValueError("Native six Adam steps disagree with joint update count")
                metrics.update({key: float(value.detach()) for key, value in native.items()})
                if any(not torch.isfinite(torch.tensor(value)) for value in self.diagnostics.values()):
                    raise ValueError("Nonfinite native optimizer diagnostics")
                metrics.update(self.diagnostics)
                if self.writer is not None:
                    for key, value in {
                        **metrics,
                        "learner_joint_updates": self.joint_updates,
                        "online_transitions": self.replay.received,
                        "sampling_rounds": self.sampling_rounds,
                        "wall_time_s": time.monotonic() - self.started,
                    }.items():
                        self.writer.add_scalar("train/ufo/" + key, value, self.joint_updates)
            if self.writer is not None:
                self.writer.flush()
            self.next_trigger += self.sample_interval
        if "action" in transitions:
            action = torch.as_tensor(transitions["action"])
            metrics.update(
                action_abs_mean=float(action.abs().mean()),
                action_saturation_fraction=float((action.abs() >= 0.99).float().mean()),
                terminated_transitions=int(torch.as_tensor(transitions["next"]["terminated"]).sum()),
                truncated_transitions=int(torch.as_tensor(transitions["next"]["truncated"]).sum()),
            )
        count = getattr(self.replay, "count", self.replay.received)
        storage = getattr(self.replay, "storage", {})
        versions = storage.get(("policy_version",))
        if versions is not None and count:
            metrics.update(
                replay_policy_age_max_updates=self.joint_updates - int(versions[:count].min()),
                replay_policy_age_min_updates=self.joint_updates - int(versions[:count].max()),
            )
        metrics.update(
            replay_draws=self.joint_updates * self.agent.cfg.train.batch_size
            if hasattr(self.agent, "cfg")
            else 0,
            replay_draw_new_ratio=self.joint_updates * self.agent.cfg.train.batch_size / self.replay.received
            if hasattr(self.agent, "cfg")
            else 0,
            replay_retained_fraction=count / self.replay.received,
        )
        return {
            **metrics,
            "learner_joint_updates": self.joint_updates,
            "collected_transitions": collected,
            "replay_size": getattr(self.replay, "count", self.replay.received),
            **{
                "expert_draws/" + source: count for source, count in getattr(self.expert, "draws", {}).items()
            },
            "online_transitions": self.replay.received,
            "sampling_rounds": self.sampling_rounds,
            "environment_control_steps": self.sampling_rounds * 4,
            "wall_time_s": time.monotonic() - self.started,
        }


def rollout_transitions(batch, *, aux_names=AUX_NAMES):
    actions = batch["actions"]
    steps, envs = actions.shape[:2]
    if actions.ndim == 4:
        actions = actions[:, :, 0]
    forward = batch["forward_inputs"]
    if not torch.equal(forward["action"].reshape(-1, 23), actions.reshape(-1, 23)):
        raise ValueError("UFO replay actions differ from behavior actions")

    def obs(values):
        return {
            k: values["states" if k == "state" else k].reshape(-1, dim).float() for k, dim in OBS_DIMS.items()
        }

    def flags(value):
        if value.shape[0] == steps + 1:
            value = value[1:]
        if value.shape[:2] != (steps, envs):
            raise ValueError("Unaligned UFO terminal flags")
        return value.reshape(-1, 1).bool()

    aux = batch["next_obs"]["ufo_aux_rewards"]
    if tuple(aux_names) not in (AUX_NAMES, VELOCITY_AUX_NAMES) or aux.shape != (steps, envs, len(aux_names)):
        raise ValueError("UFO terminal auxiliary reward layout differs")
    aux = aux.reshape(-1, len(aux_names))
    return {
        "observation": obs(batch["curr_obs"]),
        "action": actions.reshape(-1, 23).float(),
        "z": forward["z"].reshape(-1, 256).float(),
        "aux_rewards": {k: aux[:, i : i + 1] for i, k in enumerate(aux_names)},
        **{key: forward[key].reshape(-1, 1).long() for key in ("episode_id", "step_id", "policy_version")},
        "next": {
            "observation": obs(batch["next_obs"]),
            "terminated": flags(batch["terminations"]),
            "truncated": flags(batch["truncations"]),
        },
    }


def reset_control_diagnostics(transitions, *, control_profile="v1"):
    try:
        from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q
        from eb.runners.rpo_bfm_codec import policy_to_target
    except ModuleNotFoundError:
        from rpo_bfm import ACTION_GAIN, DEFAULT_Q
        from rpo_bfm_codec import policy_to_target
    initial = transitions["step_id"].reshape(-1) == 0
    count = int(initial.sum())
    if not count:
        return {"reset_control_samples": 0}
    obs = transitions["observation"]
    offset = obs["state"][initial, :23]
    residual = ((policy_to_target(obs["last_action"][initial], control_profile=control_profile)
                 - offset.new_tensor(DEFAULT_Q) - offset).abs().amax(-1))
    return {
        "reset_control_samples": count,
        "reset_projection_residual_max_rad": float(residual.max()),
        "reset_projection_residual_mean_rad": float(residual.mean()),
        "reset_projection_residual_p95_rad": float(torch.quantile(residual, 0.95)),
        "reset_projection_outside_fraction": float((residual > 1e-6).float().mean()),
    }


class UFOActorWorker(Worker):
    def __init__(self, cfg):
        Worker.__init__(self)
        self.cfg = cfg
        if self._world_size != 1:
            raise ValueError("UFO learner world_size must be one")
        self.placement = HybridComponentPlacement(cfg, Cluster())
        self.global_step = 0

    def init_worker(self):
        seed = int(self.cfg.actor.seed)
        torch.manual_seed(seed)
        import numpy as np

        np.random.seed(seed)
        cfg = self.cfg.actor.model
        agent = build_agent(
            observation_space(),
            {key: cfg[key] for key in AGENT_CONFIG_KEYS if key in cfg},
            device="cuda",
        )
        _, expert = load_robot_state(Path(cfg.motion_root), asset_sha256=cfg.asset_sha256, seed=seed,
                                    expected_codec_sha256=cfg.get("codec_sha256", expert_codec_hash(cfg)))
        if any(not windows for windows in expert.windows[250].values()):
            raise ValueError("Native rollout requires eligible long windows from every expert source")
        self.aux_names = tuple(agent.cfg.aux_rewards)
        replay = UFOReplayBridge(capacity=int(cfg.replay_capacity), seed=seed, aux_names=self.aux_names)
        self.learner = UFOLearner(agent, replay, expert, target=int(os.environ["EB_UFO_TARGET_UPDATES"]),
            sample_interval=24576 if self.cfg.actor.model.get("scaling_profile") == "env6144_batch2048_v6" else 1024,
            optimizer_profile=cfg.get("optimizer_profile"))
        from torch.utils.tensorboard import SummaryWriter

        self.learner.writer = SummaryWriter(str(Path(self.cfg.runner.logger.log_path) / "native-tensorboard"))
        self.rollout_states = {}
        self.contact_gate_records = []
        self.contact_gate_start = 0
        self.contact_last_context = None

    def set_global_step(self, global_step):
        self.global_step = global_step

    async def recv_rollout_trajectories(self, input_channel):
        count = compute_split_num(
            self.placement.get_world_size("env") * self.cfg.rollout.pipeline_stage_num, 1
        )
        trajectories = [await input_channel.get(async_op=True).async_wait() for _ in range(count)]
        self.rollout_batch = convert_trajectories_to_batch(trajectories)

    def compute_advantages_and_returns(self):
        self.transitions = rollout_transitions(self.rollout_batch, aux_names=self.aux_names)
        return {"ufo/received_transitions": len(self.transitions["action"])}

    def run_training(self):
        before=self.learner.joint_updates
        metrics = self.learner.train_round(self.transitions)
        if self.cfg.actor.model.get("learning_profile") in {
            "physical_limits_v2",
            "physical_limits_stable_latent_v3",
            "physical_limits_stable_slew_v4",
        }:
            import numpy as np

            raw_trace = self.rollout_batch["next_obs"]["ufo_action_trace"]
            width = 131 if (self.cfg.actor.model.get("ground_profile") or self.cfg.actor.model.get("physics_profile") in {"native_mjcf_v12", "native_mjcf_viability_v13"}) else 128 if self.cfg.actor.model.get("contact_profile") else 118
            if raw_trace.shape[-1] != width:
                raise ValueError("Physical action trace incomplete/nonfinite")
            trace = raw_trace.detach().cpu().numpy().reshape(-1, width)
            if len(trace) != len(self.transitions["action"]) or not np.isfinite(trace).all():
                raise ValueError("Physical action trace incomplete/nonfinite")
            np.testing.assert_array_equal(trace[..., :23], self.transitions["action"].detach().cpu().numpy())
            root = Path(self.cfg.runner.logger.log_path) / "action-traces"
            root.mkdir(exist_ok=True)
            np.savez_compressed(
                root / f"round_{self.learner.sampling_rounds}.npz",
                values=trace,
                episode_id=self.transitions["episode_id"].cpu().numpy(),
                step_id=self.transitions["step_id"].cpu().numpy(),
            )
            metrics.update(
                requested_action_abs_mean=float(np.abs(trace[..., :23]).mean()),
                applied_action_abs_mean=float(np.abs(trace[..., 23:46]).mean()),
                requested_action_saturation_fraction=float((np.abs(trace[..., :23]) >= 0.99).mean()),
                applied_action_saturation_fraction=float((np.abs(trace[..., 23:46]) >= 0.99).mean()),
                requested_applied_max_diff=float(np.abs(trace[..., :23] - trace[..., 23:46]).max()),
                hard_position_count=int(trace[..., 115].sum()),
                hard_velocity_count=int(trace[..., 116].sum()),
                hard_torque_count=int(trace[..., 117].sum()),
            )
        if self.cfg.actor.model.get("contact_profile"):
            start = self.transitions["step_id"].cpu().numpy().reshape(-1) == 0
            term = self.transitions["next"]["terminated"].cpu().numpy().reshape(-1)
            metrics.update(first_step_samples=int(start.sum()), first_step_terminated=int(term[start].sum()),
                contact_termination_count=int(trace[:,122].sum()), height_termination_count=int(trace[:,123].sum()),
                orientation_termination_count=int(trace[:,124].sum()), physics_peak_joint_velocity=float(trace[:,126].max()),
                foot_contact_peak_N=float(trace[:,118:120].max()), reset_root_lift_max_m=float(trace[:,125].max()))
        if self.cfg.actor.model.get("ground_profile"):
            metrics.update(ground_foot_margin_min_m=float(trace[:,130].min()),
                ground_outside_count=int((trace[:,130] <= 0).sum()))
            if metrics["ground_outside_count"]:
                raise ValueError("Online foot collision envelope left frozen ground")
        if self.cfg.actor.model.get("context_profile"):
            from rpo_bfm import SLEW_CONTROL
            from rpo_ufo_isaac import INTERVAL_TRACE_WIDTH
            interval=self.rollout_batch["next_obs"]["ufo_interval_trace"].detach().cpu().numpy().reshape(-1,INTERVAL_TRACE_WIDTH)
            reference=self.rollout_batch["curr_obs"]["ufo_reset_reference"].detach().cpu().numpy().reshape(-1,3)
            context=self.rollout_batch["forward_inputs"]["ufo_tracking_trace"].detach().cpu().numpy().reshape(-1,8)
            if len(interval)!=len(trace) or not np.isfinite(interval).all() or (interval[:,164]!=8).any():
                raise ValueError("Incomplete tracking/contact interval trace")
            rated=np.array(SLEW_CONTROL["joint_velocity_limits_rad_s"])
            peaks, penalties=interval[:,:23], interval[:,115:138]
            missed=((peaks>rated+.001)&(penalties<=0)).any(-1)
            if missed.any():
                raise ValueError("Substep velocity reward coverage missing")
            ended=(self.transitions["next"]["terminated"]|self.transitions["next"]["truncated"]).cpu().numpy().reshape(-1)
            tracking=context[:,0].astype(bool)
            metrics.update(strict_shadow_episode_len=float(interval[ended,163].mean()) if ended.any() else 0.,
                velocity_reward_coverage_miss_count=int(missed.sum()),
                strict_velocity_exposure_fraction=float(interval[:,138:161].mean()),
                strict_velocity_transition_fraction=float((peaks>rated+.001).any(-1).mean()),
                physics_peak_normalized_velocity=float((peaks/rated).max()),
                training_velocity_count=int(interval[:,169].sum()),
                latched_velocity_terms_without_boundary_violation=int(((interval[:,166]>0)&~(np.abs(trace[:,69:92])>rated+.001).any(-1)).sum()),
                tracking_unique_env_fraction=float(tracking.mean()),
                tracking_q_rmse_rad=float(context[tracking,7].mean()),
                tracking_windows_completed=int((tracking&(context[:,1]==249)).sum()),
                tracking_episode_starts=int((tracking&start).sum()),
                strict_shadow_tracking_episode_len=float(interval[ended&tracking,163].mean()) if (ended&tracking).any() else 0.,
                strict_shadow_mixed_episode_len=float(interval[ended&~tracking,163].mean()) if (ended&~tracking).any() else 0.)
            np.savez_compressed(root/f"interval_{self.learner.sampling_rounds}.npz",values=interval,
                context=context,reference=reference,episode_id=self.transitions["episode_id"].cpu().numpy(),
                step_id=self.transitions["step_id"].cpu().numpy())
        if self.cfg.actor.model.get("learning_profile") == "physical_limits_stable_slew_v4":
            metrics.update(reset_control_diagnostics(self.transitions, control_profile=self.cfg.actor.model.get("control_profile", "v1")))
        if self.cfg.actor.model.get('dynamics_profile')=='contact_compliance_v11':
            z=self.transitions['z'].detach().cpu().numpy().reshape(4,-1,256)
            episodes=self.transitions['episode_id'].cpu().numpy().reshape(4,-1)
            steps=self.transitions['step_id'].cpu().numpy().reshape(4,-1)
            contexts=context.reshape(4,-1,8)
            errors=0
            previous=self.contact_last_context
            for ids,times,ctx,latent in zip(episodes,steps,contexts,z):
                mask=ctx[:,0].astype(bool)
                errors+=int((ctx[mask,1]!=times[mask]-ctx[mask,6]).sum())
                if previous is not None:
                    old_ids,old_times,old_ctx,old_z=previous
                    same=(ids==old_ids)&(times==old_times+1)
                    steady=same&(~mask)&(times%100!=0)
                    errors+=int((latent[steady]!=old_z[steady]).any(-1).sum())
                    tracking=same&mask&(ctx[:,1]!=0)
                    errors+=int((ctx[tracking,2:7]!=old_ctx[tracking,2:7]).any(-1).sum())
                previous=(ids.copy(),times.copy(),ctx.copy(),latent.copy())
            self.contact_last_context=previous
            if errors:raise ValueError('Unexpected within-episode tracking or mixed context changes')
            metrics['context_error_count']=errors
            updates=self.learner.joint_updates-before
            for step in range(before+1,self.learner.joint_updates+1):
                self.contact_gate_records.append({'step':step,
                    'strict_shadow_length_sum':float(interval[ended,163].sum())/updates,
                    'completed_episodes':int(ended.sum())/updates,
                    'physics_max_rated_ratio':metrics['physics_peak_normalized_velocity'],
                    'first_terminated':metrics['first_step_terminated']/updates,
                    'first_samples':metrics['first_step_samples']/updates,
                    'coverage_miss':metrics['velocity_reward_coverage_miss_count'],
                    'ground_outside':metrics['ground_outside_count'],'context_errors':errors})
            metrics.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return {"ufo/" + key: value for key, value in metrics.items()}

    async def sync_model_to_rollout(self):
        from omegaconf import OmegaConf
        from rlinf.hybrid_engines.weight_syncer import WeightSyncer

        options = WeightSyncer.create(OmegaConf.select(self.cfg, "weight_syncer")).comm_options
        state = self.learner.agent._model.state_dict()
        packet = {
            "model": {k: v.detach().cpu().clone() for k, v in state.items() if not k.startswith("_target_")},
            "pool": pool_state(self.learner.agent),
            "version": self.learner.joint_updates,
            "online_transitions": self.learner.replay.received,
        }
        ranks = list(range(self.placement.get_world_size("rollout")))
        self.sync_expected = sync_identity(packet)
        await self.broadcast(
            packet,
            groups=[(self._group_name, 0), (self.cfg.rollout.group_name, ranks)],
            src=(self._group_name, 0),
            async_op=True,
            options=options,
        ).async_wait()

    def progress(self):
        return {
            "learner_joint_updates": self.learner.joint_updates,
            "online_transitions": self.learner.replay.received,
            "sampling_rounds": self.learner.sampling_rounds,
            "target_reached": self.learner.joint_updates >= self.learner.target,
        }

    def set_rollout_states(self, states):
        count = self.placement.get_world_size("rollout")
        if len(states) != count:
            raise ValueError("UFO checkpoint needs every sampler rank state")
        self.rollout_states = {f"rank_{i}": state for i, state in enumerate(states)}

    def verify_sampler_sync(self, receipts):
        validate_sampler_identities(
            self.sync_expected, receipts, ranks=self.placement.get_world_size("rollout")
        )
        from rlinf_entry import atomic_json

        atomic_json(
            Path(self.cfg.runner.logger.log_path) / "ufo-sampler-sync.json",
            {
                "schema_version": "eb/ufo-sampler-sync/v1",
                "identity": self.sync_expected,
                "samplers": receipts,
                "sampling_rounds": self.learner.sampling_rounds,
            },
        )
        return self.sync_expected

    def restored_rollout_states(self):
        return self.rollout_states

    def save_checkpoint(self, save_path: str, step: int = 0):
        import json

        from bfm_ufo_checkpoint import save_ufo_checkpoint, ufo_checkpoint_root

        if self.cfg.actor.model.get("learning_profile") in {
            "physical_limits_stable_latent_v3",
            "physical_limits_stable_slew_v4",
        }:
            save_path = (
                ufo_checkpoint_root(Path(self.cfg.runner.logger.log_path), self.cfg) / Path(save_path).name
            )

        receipt = {
            **json.loads(os.environ["EB_INPUT_RECEIPT"]),
            **self.progress(),
            "run_id": os.environ["EB_RUN_ID"],
            "compatibility_fingerprint": os.environ["EB_RESUME_FINGERPRINT"],
            "next_trigger": self.learner.next_trigger,
        }
        if self.cfg.actor.model.get('dynamics_profile')=='contact_compliance_v11':
            from rpo_ufo_compliance import long_training_gate
            receipt['training_gate']=long_training_gate(self.contact_gate_records,
                start=self.contact_gate_start,target=self.learner.joint_updates)
            receipt['training_gate_records']=self.contact_gate_records
        save_ufo_checkpoint(
            self.learner.agent,
            {"train": self.learner.replay, "expert_slicer": self.learner.expert},
            self.rollout_states,
            Path(save_path),
            receipt=receipt,
        )

    def load_checkpoint(self, load_path: str):
        from bfm_ufo_checkpoint import load_ufo_checkpoint

        result = load_ufo_checkpoint(
            Path(load_path),
            agent=self.learner.agent,
            replay={"train": self.learner.replay, "expert_slicer": self.learner.expert},
            expected={"compatibility_fingerprint": os.environ["EB_RESUME_FINGERPRINT"]},
            device="cuda",
        )
        self.learner.joint_updates = result["receipt"]["learner_joint_updates"]
        self.learner.sampling_rounds = result["receipt"]["sampling_rounds"]
        self.learner.next_trigger = result["receipt"]["next_trigger"]
        self.rollout_states = result["rollout"]
        self.contact_gate_start = self.learner.joint_updates
