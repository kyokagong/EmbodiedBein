"""Boundary to pinned UFO; v7 declares bounded regularization and Adam gradients."""

from __future__ import annotations

import re

UFO_REVISION = "0b89378de02f77712072d983bc43ec2a46455dc0"
OBS_DIMS = {"state": 52, "privileged_state": 358, "last_action": 23, "history_actor": 300}
AGENT_CONFIG_KEYS = (
    "hidden_dim", "hidden_layers", "batch_size", "rollout_expert_trajectories", "architecture_profile",
    "auxiliary_profile", "optimizer_profile",
)
OPTIMIZERS = (
    "forward_optimizer",
    "backward_optimizer",
    "discriminator_optimizer",
    "actor_optimizer",
    "critic_optimizer",
    "aux_critic_optimizer",
)
TARGETS = ("_target_forward_map", "_target_backward_map", "_target_critic", "_target_aux_critic")
AUX_NAMES = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)
VELOCITY_AUX_NAMES = (*AUX_NAMES, "limits_dof_vel")


def validate_source_receipt(receipt: dict) -> dict:
    if receipt.get("schema_version") != "eb/ufo-image-source/v1":
        raise ValueError("Unknown UFO source receipt")
    for provider in ("ufo", "rlinf"):
        source = receipt.get(provider) or {}
        if (
            not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(source.get("content_hash", "")))
            or not source.get("license")
        ):
            raise ValueError(f"Missing {provider} revision/content/license")
    if receipt["ufo"]["commit"] != UFO_REVISION or receipt["ufo"]["license"] != "CC-BY-NC-4.0":
        raise ValueError("UFO source differs from approved revision/license")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("patch_hash", ""))):
        raise ValueError("Missing source patch hash")
    packages = receipt.get("packages") or {}
    if any(
        not packages.get(name)
        for name in ("python", "torch", "gymnasium", "pydantic", "safetensors", "tensordict")
    ):
        raise ValueError("Missing actual agent dependency versions")
    if not packages["torch"].startswith("2.7."):
        raise ValueError("Unverified UFO Torch version")
    return receipt


def to_ufo_observation(obs: dict) -> dict:
    if "state" in obs and "states" in obs:
        raise ValueError("Ambiguous state observation")
    return {("state" if key == "states" else key): value for key, value in obs.items()}


def agent_config(cfg: dict, *, device: str):
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    from humanoidverse.agents.presets.fb import build_fb_agent

    if set(cfg) - set(AGENT_CONFIG_KEYS):
        raise ValueError("Unknown UFO agent configuration keys")
    tracking = cfg.get("rollout_expert_trajectories", True)
    if type(tracking) is not bool:
        raise ValueError("UFO rollout_expert_trajectories must be boolean")
    values = build_fb_agent(device=device, compile=False).model_dump()
    profile = cfg.get("architecture_profile")
    if profile == "native_fb_v1":
        if cfg.get("hidden_dim", 2048) != 2048 or cfg.get("hidden_layers", 6) != 6:
            raise ValueError("Native UFO architecture conflicts with requested sizes")
    elif profile is None:
        for name in ("f", "actor", "critic", "aux_critic", "discriminator"):
            values["model"]["archi"][name].update(
                hidden_dim=int(cfg.get("hidden_dim", 1024)), hidden_layers=int(cfg.get("hidden_layers", 3))
            )
    else:
        raise ValueError("Unknown UFO architecture profile")
    auxiliary = cfg.get("auxiliary_profile")
    if auxiliary is not None:
        if auxiliary not in {"soft_velocity_v5", "soft_velocity_v7"} or profile != "native_fb_v1":
            raise ValueError("Unknown or incompatible UFO auxiliary profile")
        if values["aux_rewards"] != list(AUX_NAMES) or set(values["aux_rewards_scaling"]) != set(AUX_NAMES):
            raise ValueError("Native UFO auxiliary preset differs from frozen eight rewards")
        values["aux_rewards"] = list(VELOCITY_AUX_NAMES)
        values["aux_rewards_scaling"]["limits_dof_vel"] = -25.0 if auxiliary == "soft_velocity_v7" else -5.0
    optimizer_profile = cfg.get("optimizer_profile")
    if optimizer_profile is not None:
        if optimizer_profile != "stabilized_v7" or auxiliary != "soft_velocity_v7":
            raise ValueError("Unknown or incompatible UFO optimizer profile")
        values["train"]["lr_discriminator"] = 3e-6
        # Clip all six optimizers once at their Adam boundary, including both critics.
        values["train"]["clip_grad_norm"] = 0
    values["train"]["batch_size"] = int(cfg.get("batch_size", 1024 if profile else 128))
    values["train"]["rollout_expert_trajectories"] = tracking
    return FBcprAuxAgentConfig.model_validate(values)


def build_agent(obs_space, cfg: dict, *, device: str):
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError("Native gradient collective must contain exactly one learner")
    if {name: space.shape for name, space in obs_space.spaces.items()} != {
        name: (dim,) for name, dim in OBS_DIMS.items()
    }:
        raise ValueError("UFO Isaac observation schema differs from v1 codec")
    return agent_config(cfg, device=device).build(obs_space, action_dim=23)


def optimizer_steps(agent) -> dict[str, int]:
    steps = {}
    for name in OPTIMIZERS:
        values = {int(item["step"].item()) for item in getattr(agent, name).state.values()}
        if len(values) > 1:
            raise ValueError(f"Inconsistent Adam steps in {name}")
        steps[name] = next(iter(values), 0)
    return steps


def instrument_updates(agent, *, optimizer_profile=None) -> dict[str, float]:
    """Observe Adam boundaries; the declared v7 profile clips before each step."""
    import torch

    if optimizer_profile not in {None, "stabilized_v7"}:
        raise ValueError("Unknown optimizer instrumentation profile")
    metrics = {}
    for name in OPTIMIZERS:
        optimizer = getattr(agent, name)
        before = []

        def pre(opt, args, kwargs, *, label=name, saved=before):
            params = [p for group in opt.param_groups for p in group["params"] if p.grad is not None]
            saved[:] = [(p, p.detach().clone()) for p in params]
            metrics[label + "/lr"] = float(opt.param_groups[0]["lr"])
            metrics[label + "/grad_norm_l2"] = float(
                torch.stack([p.grad.detach().square().sum() for p in params]).sum().sqrt()
            )
            if optimizer_profile == "stabilized_v7":
                limit = 1.0 if label == "actor_optimizer" else 5.0
                torch.nn.utils.clip_grad_norm_(params, limit, error_if_nonfinite=True)
                metrics[label + "/grad_norm_after_clip_l2"] = float(
                    torch.stack([p.grad.detach().square().sum() for p in params]).sum().sqrt()
                )
                metrics[label + "/grad_clip_limit"] = limit

        def post(opt, args, kwargs, *, label=name, saved=before):
            metrics[label + "/parameter_delta_l2"] = float(
                torch.stack([(p.detach() - value).square().sum() for p, value in saved]).sum().sqrt()
            )
            saved.clear()

        optimizer.register_step_pre_hook(pre)
        optimizer.register_step_post_hook(post)
    return metrics
