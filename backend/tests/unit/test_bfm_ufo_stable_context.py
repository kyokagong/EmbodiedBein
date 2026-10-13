import pytest

torch = pytest.importorskip("torch")

from eb.runners.bfm_ufo_agent import agent_config  # noqa: E402
from eb.runners.bfm_ufo_policy import UFOBFMPolicy  # noqa: E402


def test_native_tracking_option_preserves_default_and_accepts_false():
    assert agent_config({}, device="cpu").train.rollout_expert_trajectories is True
    config = agent_config({"rollout_expert_trajectories": False}, device="cpu")
    assert config.train.rollout_expert_trajectories is False
    assert config.train.use_mix_rollout is True
    assert config.train.expert_asm_ratio == .6
    assert config.model.actor_std == .05


def test_native_architecture_matches_pinned_preset_without_enlarging_discriminator():
    from humanoidverse.agents.presets.fb import build_fb_agent
    config = agent_config({"architecture_profile": "native_fb_v1",
                           "rollout_expert_trajectories": False}, device="cpu")
    preset = build_fb_agent(device="cpu", compile=False)
    assert config.model.model_dump() == preset.model.model_dump()
    assert config.train.batch_size == preset.train.batch_size == 1024
    assert config.model.archi.discriminator.hidden_dim == 1024
    assert config.model.archi.b.hidden_dim == 256


@pytest.mark.parametrize("config", [
    {"architecture_profile": "unknown"},
    {"architecture_profile": "native_fb_v1", "hidden_dim": 1024},
    {"architecture_profile": "native_fb_v1", "hidden_layers": 3},
])
def test_native_architecture_rejects_conflicting_sizes(config):
    with pytest.raises(ValueError, match="architecture"):
        agent_config(config, device="cpu")


@pytest.mark.parametrize("value", [0, "false", None])
def test_native_tracking_option_rejects_nonboolean(value):
    with pytest.raises(ValueError, match="boolean"):
        agent_config({"rollout_expert_trajectories": value}, device="cpu")


def test_one_reset_preserves_other_native_sampler_context_without_tracking(monkeypatch):
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16,
                          "rollout_expert_trajectories": False})
    assert policy.native.cfg.train.rollout_expert_trajectories is False
    def unexpected_tracking(*args, **kwargs):
        raise AssertionError("stable rollout must not sample tracking sequences")
    monkeypatch.setattr(policy.native, "_sample_tracking_z", unexpected_tracking)
    obs = {"states": torch.zeros(2, 52), "privileged_state": torch.zeros(2, 358),
           "last_action": torch.zeros(2, 23), "history_actor": torch.zeros(2, 300),
           "episode_id": torch.tensor([[0], [1]]), "step_id": torch.tensor([[2], [2]])}
    policy.predict_action_batch(env_obs=obs)
    before = policy.z.clone()
    obs["episode_id"][0] = 2
    obs["step_id"][:] = torch.tensor([[0], [3]])
    policy.predict_action_batch(env_obs=obs)
    torch.testing.assert_close(policy.z[1], before[1], rtol=0, atol=0)
    assert not torch.equal(policy.z[0], before[0])
    obs["step_id"][:] = torch.tensor([[99], [4]])
    policy.predict_action_batch(env_obs=obs)
    before = policy.z.clone()
    obs["step_id"][:] = torch.tensor([[100], [5]])
    policy.predict_action_batch(env_obs=obs)
    assert not torch.equal(policy.z[0], before[0])
    torch.testing.assert_close(policy.z[1], before[1], rtol=0, atol=0)

