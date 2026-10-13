import copy

import pytest

torch = pytest.importorskip("torch")

from eb.runners.bfm_ufo_agent import AUX_NAMES, agent_config  # noqa: E402
from eb.runners.rpo_ufo_isaac import raw_aux_rewards  # noqa: E402


def inputs():
    batch = 4
    data = {
        "q": torch.zeros(batch, 23),
        "torques": torch.zeros(batch, 23),
        "joint_limits": torch.tensor([-2.0, 2.0]).repeat(batch, 23, 1),
        "torque_limits": torch.ones(batch, 23) * 100,
        "body_quat_w": torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(batch, 24, 1),
        "body_lin_vel_w": torch.zeros(batch, 24, 3),
        "dq": torch.zeros(batch, 23),
        "velocity_limits": torch.tensor([25.0] * 5 + [8.0] * 18),
    }
    contact = {"forces": torch.zeros(batch, 24, 3), "body_names": [str(i) for i in range(24)]}
    mapping = {"feet": [1, 2], "contact_feet": [1, 2], "ankles": [3, 4], "undesired_contacts": [0]}
    return data, contact, mapping


def test_velocity_aux_is_smooth_dimensionless_and_preserves_eight_original_values():
    data, contact, mapping = inputs()
    data["dq"][:, 0] = torch.tensor([19.0, 20.0, 25.0, -25.0])
    action = torch.zeros(4, 23)
    old = raw_aux_rewards(data, action, action, contact, mapping)
    new = raw_aux_rewards(data, action, action, contact, mapping, aux_names=(*AUX_NAMES, "limits_dof_vel"))
    torch.testing.assert_close(new[:, :8], old, atol=0, rtol=0)
    torch.testing.assert_close(new[:, 8], torch.tensor([0.0, 0.0, 0.04, 0.04]), atol=1e-7, rtol=1e-6)
    dq = data["dq"].clone().requires_grad_()
    data["dq"] = dq
    raw_aux_rewards(data, action, action, contact, mapping, aux_names=(*AUX_NAMES, "limits_dof_vel"))[
        :, -1
    ].sum().backward()
    torch.testing.assert_close(dq.grad[:, 0], torch.tensor([0.0, 0.0, 0.016, -0.016]), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("bad", [0.0, float("nan")])
def test_velocity_aux_rejects_invalid_rated_velocity(bad):
    data, contact, mapping = inputs()
    data["velocity_limits"][0] = bad
    with pytest.raises(ValueError, match="velocity"):
        raw_aux_rewards(
            data,
            torch.zeros(4, 23),
            torch.zeros(4, 23),
            contact,
            mapping,
            aux_names=(*AUX_NAMES, "limits_dof_vel"),
        )


def test_soft_velocity_uses_native_aux_interface_without_changing_networks_or_other_weights():
    cfg = {
        "architecture_profile": "native_fb_v1",
        "hidden_dim": 2048,
        "hidden_layers": 6,
        "batch_size": 1024,
        "rollout_expert_trajectories": False,
    }
    baseline = agent_config(cfg, device="cpu").model_dump()
    enhanced = agent_config({**cfg, "auxiliary_profile": "soft_velocity_v5"}, device="cpu").model_dump()
    assert enhanced["aux_rewards"] == [*AUX_NAMES, "limits_dof_vel"]
    assert enhanced["aux_rewards_scaling"]["limits_dof_vel"] == -5.0
    restored = copy.deepcopy(enhanced)
    restored["aux_rewards"].pop()
    restored["aux_rewards_scaling"].pop("limits_dof_vel")
    assert restored == baseline
    with pytest.raises(ValueError, match="auxiliary"):
        agent_config({**cfg, "auxiliary_profile": "custom"}, device="cpu")


def velocity_batch():
    from eb.runners.bfm_ufo_agent import OBS_DIMS

    current = {("states" if key == "state" else key): torch.zeros(4, 2, dim) for key, dim in OBS_DIMS.items()}
    terminal = {key: value + 7 for key, value in current.items()}
    terminal["ufo_aux_rewards"] = torch.zeros(4, 2, 9)
    terminal["ufo_aux_rewards"][..., -1] = 0.04
    action = torch.full((4, 2, 1, 23), 0.2)
    flags = torch.zeros(5, 2, 1, dtype=torch.bool)
    flags[-1, 0] = True
    return {
        "actions": action,
        "curr_obs": current,
        "next_obs": terminal,
        "forward_inputs": {
            "action": action[:, :, 0],
            "z": torch.ones(4, 2, 256),
            **{
                key: torch.zeros(4, 2, 1, dtype=torch.long)
                for key in ("episode_id", "step_id", "policy_version")
            },
        },
        "terminations": flags,
        "truncations": torch.zeros_like(flags),
    }


def test_ninth_terminal_aux_survives_transition_and_replay_restore():
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.bfm_ufo_worker import rollout_transitions

    names = (*AUX_NAMES, "limits_dof_vel")
    batch = velocity_batch()
    transitions = rollout_transitions(batch, aux_names=names)
    assert transitions["next"]["terminated"][-2:].tolist() == [[True], [False]]
    replay = UFOReplayBridge(capacity=8, aux_names=names)
    replay.append(transitions)
    restored = UFOReplayBridge(capacity=8, aux_names=names)
    restored.load_state_dict(replay.state_dict())
    sample = restored.sample(16)
    torch.testing.assert_close(sample["aux_rewards"]["limits_dof_vel"], torch.full((16, 1), 0.04))
    assert (sample["next"]["observation"]["state"] == 7).all()
    with pytest.raises(ValueError, match="auxiliary"):
        rollout_transitions(batch)
    batch["next_obs"]["ufo_aux_rewards"] = torch.zeros(4, 2, 8)
    with pytest.raises(ValueError, match="auxiliary"):
        rollout_transitions(batch, aux_names=names)
    transitions["aux_rewards"].pop("limits_dof_vel")
    with pytest.raises(ValueError):
        replay.append(transitions)


@pytest.mark.parametrize("populated", [False, True])
def test_velocity_replay_identity_rejects_cross_profile_even_when_empty(populated):
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.bfm_ufo_worker import rollout_transitions

    names = (*AUX_NAMES, "limits_dof_vel")
    enhanced = UFOReplayBridge(capacity=8, aux_names=names)
    baseline = UFOReplayBridge(capacity=8)
    if populated:
        enhanced.append(rollout_transitions(velocity_batch(), aux_names=names))
    for destination, source in ((baseline, enhanced), (enhanced, baseline)):
        with pytest.raises(ValueError):
            destination.load_state_dict(source.state_dict())


def test_native_update_consumes_ninth_raw_aux_once_and_keeps_six_adam(tmp_path, monkeypatch):
    import numpy as np
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig

    from eb.runners.bfm_ufo_agent import optimizer_steps
    from eb.runners.bfm_ufo_policy import observation_space
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer, UFOReplayBridge
    from eb.runners.bfm_ufo_worker import rollout_transitions

    torch.set_num_threads(1)
    values = agent_config(
        {
            "architecture_profile": "native_fb_v1",
            "auxiliary_profile": "soft_velocity_v5",
            "rollout_expert_trajectories": False,
        },
        device="cpu",
    ).model_dump()
    for architecture in values["model"]["archi"].values():
        if isinstance(architecture, dict) and "hidden_dim" in architecture:
            architecture.update(hidden_dim=64, hidden_layers=2)
    values["train"]["batch_size"] = 16
    agent = FBcprAuxAgentConfig.model_validate(values).build(observation_space(), action_dim=23)
    weighted = []
    agent._model._aux_reward_normalizer.register_forward_pre_hook(
        lambda module, args: weighted.append(args[0].detach().clone())
    )
    replay = UFOReplayBridge(capacity=8, aux_names=tuple(values["aux_rewards"]))
    replay.append(rollout_transitions(velocity_batch(), aux_names=replay.aux_names))
    expert = UFOExpertSlicer(
        [
            {
                "id": source,
                "source": source,
                "source_group": source,
                "split": "train",
                "arrays": {
                    "states": np.zeros((270, 52), np.float32),
                    "privileged_state": np.zeros((270, 358), np.float32),
                    "valid": np.ones(270, bool),
                },
            }
            for source in ("official_rpo", "g1")
        ],
        manifest_sha256="a" * 64,
    )
    metrics = agent.update({"train": replay, "expert_slicer": expert}, 11264)
    assert set(optimizer_steps(agent).values()) == {1}
    torch.testing.assert_close(metrics["aux_rew/limits_dof_vel"], torch.tensor(0.04))
    assert len(weighted) == 1
    torch.testing.assert_close(weighted[0], torch.full((16, 1), -0.2))
    assert all(torch.isfinite(value).all() for value in metrics.values())

    from eb.domain.training_recipe import RECIPES, UFO_VELOCITY_RECIPE
    from eb.runners import bfm_ufo_policy
    from eb.runners.bfm_ufo_checkpoint import load_ufo_checkpoint, save_ufo_checkpoint

    native_config = FBcprAuxAgentConfig.model_validate(values)
    monkeypatch.setattr(bfm_ufo_policy, "agent_config", lambda cfg, device: native_config)
    ranks = {
        f"rank_{rank}": bfm_ufo_policy.UFOBFMPolicy({"seed": 42 + rank}).rollout_state_dict()
        for rank in range(3)
    }
    checkpoint = tmp_path / "global_step_1"
    save_ufo_checkpoint(
        agent,
        {"train": replay, "expert_slicer": expert},
        ranks,
        checkpoint,
        receipt={
            "run_id": "run_fixture",
            "learner_joint_updates": 1,
            "sampling_rounds": 1,
            "online_transitions": 8,
            "next_trigger": 11264,
            "compatibility_fingerprint": "sha256:" + "b" * 64,
            "execution": RECIPES[UFO_VELOCITY_RECIPE].binding.model_dump(),
        },
    )
    expected = agent.update({"train": replay, "expert_slicer": expert}, 12288)
    restored = native_config.build(observation_space(), action_dim=23)
    buffers = {
        "train": UFOReplayBridge(capacity=8, aux_names=tuple(values["aux_rewards"])),
        "expert_slicer": expert,
    }
    load_ufo_checkpoint(
        checkpoint,
        agent=restored,
        replay=buffers,
        expected={"compatibility_fingerprint": "sha256:" + "b" * 64},
        device="cpu",
    )
    actual = restored.update(buffers, 12288)
    assert set(optimizer_steps(restored).values()) == {2}
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, atol=1e-6, rtol=1e-5)
    for key, value in agent._model.state_dict().items():
        torch.testing.assert_close(restored._model.state_dict()[key], value, atol=1e-6, rtol=1e-5)


def test_velocity_observation_bootstraps_nine_columns_before_reference_initialization(monkeypatch):
    import sys
    from types import SimpleNamespace

    from eb import rpo_bfm
    from eb.runners import rpo_ufo_isaac

    observed = []

    class BootstrapObserved(Exception):
        pass

    class Base:
        def __init__(self):
            self.num_envs, self.device = 2, "cpu"
            self.bfm_last_action = torch.zeros(2, 23)
            self.robot = SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos_target=torch.zeros(2, 23),
                    joint_vel=torch.zeros(2, 23),
                    applied_torque=torch.zeros(2, 23),
                )
            )
            observed.append(self._get_observations())
            raise BootstrapObserved

        def _get_observations(self):
            return {}

    monkeypatch.setitem(sys.modules, "rpo_bfm", rpo_bfm)
    monkeypatch.setitem(sys.modules, "rpo_bfm_extension", SimpleNamespace(RPOBFMIsaacEnv=Base))
    monkeypatch.setitem(sys.modules, "rpo_flat_extension", SimpleNamespace(FlatResetCompat=object))
    monkeypatch.setenv("EB_UFO_LEARNING_PROFILE", "physical_limits_slew_v4")
    monkeypatch.setenv("EB_UFO_AUXILIARY_PROFILE", "soft_velocity_v5")
    with pytest.raises(BootstrapObserved):
        rpo_ufo_isaac.__getattr__("RPOUFOIsaacPhysics")()
    assert observed[0]["ufo_aux_rewards"].shape == (2, 9)
    assert observed[0]["ufo_action_trace"].shape == (2, 118)


def test_velocity_evaluation_freezes_auxiliary_profile(monkeypatch):
    import os

    from eb.rpo_bfm import SLEW_CONTROL
    from eb.runners.rpo_bfm_eval import configure_action_control

    monkeypatch.delenv("EB_UFO_AUXILIARY_PROFILE", raising=False)
    exported = {
        "execution": {"recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5"},
        "action_control": SLEW_CONTROL,
        "auxiliary_profile": "soft_velocity_v5",
    }
    configure_action_control(exported)
    assert os.environ["EB_UFO_AUXILIARY_PROFILE"] == "soft_velocity_v5"
    with pytest.raises(ValueError, match="auxiliary"):
        configure_action_control({**exported, "auxiliary_profile": None})
    configure_action_control({})
    assert "EB_UFO_AUXILIARY_PROFILE" not in os.environ


def test_ninth_aux_survives_real_auto_reset_envoutput_builder_and_replay():
    from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
    from rlinf.data.schema.embodied_types import EnvOutput
    from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.bfm_ufo_worker import rollout_transitions

    batch = velocity_batch()
    current = {key: value[0] for key, value in batch["curr_obs"].items()}
    final = {key: value[0] for key, value in batch["next_obs"].items()}
    final["ufo_action_trace"] = torch.ones(2, 118)
    reset = {key: value.clone() for key, value in final.items()}
    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs = 2

    def reset_done(*, env_ids):
        for value in reset.values():
            value[env_ids] = 0
        return reset, {}

    env.reset = reset_done
    done = torch.tensor([True, False])
    obs, infos = env._handle_auto_reset(done, final, {})
    output = EnvOutput(obs=obs, final_obs=infos["final_observation"], dones=done).to_dict()
    assert output["obs"]["ufo_aux_rewards"][0].sum() == 0
    builder = EmbodiedTrajectoryBuilder()
    builder.append_transitions(current, output["final_obs"])
    trajectory = builder.to_trajectory()
    batch.update(
        actions=batch["actions"][:1],
        curr_obs=trajectory.curr_obs,
        next_obs=trajectory.next_obs,
        forward_inputs={key: value[:1] for key, value in batch["forward_inputs"].items()},
        terminations=torch.stack((torch.zeros_like(done), done)).unsqueeze(-1),
        truncations=torch.zeros(2, 2, 1, dtype=torch.bool),
    )
    names = (*AUX_NAMES, "limits_dof_vel")
    replay = UFOReplayBridge(capacity=2, aux_names=names)
    replay.append(rollout_transitions(batch, aux_names=names))
    sampled = replay.sample(16)
    torch.testing.assert_close(sampled["aux_rewards"]["limits_dof_vel"], torch.full((16, 1), 0.04))
    assert (sampled["next"]["observation"]["state"] == 7).all()
