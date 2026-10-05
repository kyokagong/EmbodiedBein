"""CPU checks inside the RPO task image; no simulator or training launch."""

from pathlib import Path
import importlib.util

import torch

from rpo_flat_policy import RPOFlatPolicy
from rpo_flat_worker import AdaptiveKLScheduler, adaptive_kl_lr, clipped_value_mse
from rpo_flat_extension import FlatResetCompat


def main():
    torch.manual_seed(42)
    policy = RPOFlatPolicy(780, 23, 1, True, False, critic_obs_dim=1390)
    assert [layer.out_features for layer in policy.backbone if isinstance(layer, torch.nn.Linear)] == [512, 256, 128]
    assert torch.equal(policy.actor_std, torch.ones(23))
    obs = {"states": torch.randn(8, 780), "critic_states": torch.randn(8, 1390)}
    actions, result = policy.predict_action_batch(obs)
    assert actions.shape == (8, 1, 23)
    assert result["prev_values"].shape == (8, 1)
    inputs = result["forward_inputs"]
    output = policy.default_forward(inputs)
    assert torch.allclose(output["logprobs"], result["prev_logprobs"])
    assert output["flat_kl"].abs().max() < 3e-4
    expected = torch.distributions.Normal(inputs["old_mean"], inputs["old_std"]).log_prob(actions[:, 0]).sum(-1)
    assert torch.allclose(result["prev_logprobs"].sum(-1), expected)
    # Old rollout inputs must remain usable after a statistics update.
    policy.update_normalizers(inputs["raw_states"], inputs["raw_critic_states"])
    assert policy.actor_norm.count.item() == 8
    assert torch.allclose(policy.default_forward(inputs)["logprobs"], result["prev_logprobs"])
    resumed = RPOFlatPolicy(780, 23, 1, True, False, critic_obs_dim=1390)
    resumed.load_state_dict(policy.state_dict(), strict=True)
    exported = torch.nn.Sequential(resumed.actor_norm, resumed.backbone, resumed.actor_mean)
    eval_actions, _ = resumed.predict_action_batch(obs, mode="eval")
    assert torch.allclose(exported(obs["states"]), eval_actions[:, 0])
    assert torch.equal(resumed.actor_norm._mean, policy.actor_norm._mean)
    try:
        policy.predict_action_batch({"states": obs["states"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Missing privileged observations accepted")
    assert clipped_value_mse(torch.tensor([[10.]]), torch.tensor([[0.]]), torch.tensor([[0.]]), .2).item() == 100
    assert adaptive_kl_lr(.001, .03, .01) == .001 / 1.5
    assert adaptive_kl_lr(.001, .001, .01) == .001 * 1.5
    assert adaptive_kl_lr(.001, 0., .01) == .001
    assert adaptive_kl_lr(1e-5, 100., .01) == 1e-5
    assert adaptive_kl_lr(.01, .001, .01) == .01
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    scheduler = AdaptiveKLScheduler(optimizer)
    optimizer.param_groups[0]["lr"] = .0002
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == .0002
    # Exercise reset ownership with the actual pinned Isaac history buffer.
    spec = importlib.util.spec_from_file_location("flat_circular", Path("/workspace/isaaclab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class OfficialStub:
        def __init__(self):
            self.device = "cpu"
            self.num_envs = 2
            self.extras = {}
            self.actor_obs_buffer = module.CircularBuffer(10, 2, "cpu")
            self.critic_obs_buffer = module.CircularBuffer(10, 2, "cpu")
            self.state = torch.tensor([[1.], [10.]])
            self._get_observations()
            self.state += 1
            self.obs_buf = self._get_observations()

        def _get_observations(self):
            self.actor_obs_buffer.append(self.state)
            self.critic_obs_buffer.append(self.state * 2)
            return {"policy": self.actor_obs_buffer.buffer.flatten(1), "critic": self.critic_obs_buffer.buffer.flatten(1)}

        def _reset_idx(self, ids):
            self.state[ids] = 0
            self.actor_obs_buffer.reset(ids)
            self.critic_obs_buffer.reset(ids)

        def step(self, actions):
            self.state += 1
            self._reset_idx(torch.tensor([0]))
            self.obs_buf = self._get_observations()
            return self.obs_buf, None, torch.tensor([True, False]), torch.tensor([True, False]), {}

    class Bridge(FlatResetCompat, OfficialStub):
        pass

    env = Bridge()
    terminal, _, termination, timeout, _ = env.step(None)
    assert terminal["policy"][0, -1] == 3
    assert termination[0] and timeout[0]
    active_history = env.actor_obs_buffer.buffer[1].clone()
    active_obs = terminal["policy"][1].clone()
    reset_obs, _ = env.reset(env_ids=torch.tensor([0]))
    assert reset_obs["policy"][0].eq(0).all()
    assert torch.equal(reset_obs["policy"][1], active_obs)
    assert torch.equal(env.actor_obs_buffer.buffer[1], active_history)
    assert env.actor_obs_buffer._num_pushes.tolist() == [1, 3]
    env.state += 1
    next_obs = env._get_observations()
    assert next_obs["policy"][1, -2:].tolist() == [12., 13.]
    print("RPO Flat policy, normalization, joint logprob, PPO hooks and partial reset checks passed")


def patched_rlinf_check():
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.mlp_policy import get_model
    from rlinf.algorithms.losses import compute_ppo_critic_loss
    from rlinf.algorithms.registry import policy_loss

    os.environ["EMBODIED_PATH"] = "/tmp/flat_rlinf/examples/embodiment"
    config_dir = Path(__file__).resolve().parents[2] / "eb/runners/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="rpo_flat_ppo")
    OmegaConf.resolve(cfg)
    assert cfg.env.train.total_num_envs == 512 and cfg.env.eval.total_num_envs == 64
    assert cfg.env.train.seed == cfg.actor.seed == 42
    assert cfg.actor.global_batch_size * 4 == 512 * 24
    assert cfg.actor.micro_batch_size * 4 == cfg.actor.global_batch_size
    assert cfg.runner.max_epochs == 500
    assert not cfg.runner.use_training_pipeline and not cfg.runner.overlap_env_bootstrap

    policy = get_model(OmegaConf.create({"model_type": "mlp_policy", "policy_setup": "rpo-v2-flat-23dof",
                                        "obs_dim": 780, "critic_obs_dim": 1390, "action_dim": 23,
                                        "num_action_chunks": 1, "add_value_head": True}))
    assert isinstance(policy, RPOFlatPolicy)
    args = {"values": torch.tensor([[10.]]), "returns": torch.tensor([[0.]]),
            "prev_values": torch.tensor([[0.]]), "value_clip": .2, "huber_delta": 10.}
    assert compute_ppo_critic_loss(**args, rpo_flat_value_mse=True)[0].item() == 100
    assert compute_ppo_critic_loss(**args)[0].item() == 50
    logprobs = torch.zeros(2, 23)
    logprobs[:, :2] = torch.tensor([.2, .3])
    args = {"loss_type": "actor_critic", "task_type": "embodied", "logprob_type": "action_level",
            "reward_type": "action_level", "single_action_dim": 23,
            "logprobs": logprobs, "old_logprobs": torch.zeros_like(logprobs),
            "values": torch.zeros(2, 1), "prev_values": torch.zeros(2, 1), "returns": torch.zeros(2, 1),
            "advantages": torch.ones(2, 1), "clip_ratio_high": .2, "clip_ratio_low": .2,
            "value_clip": .2, "huber_delta": 10., "rpo_flat_value_mse": True}
    _, metrics = policy_loss(**args)
    assert abs(metrics["actor/ratio"] - torch.exp(torch.tensor(.5)).item()) < 1e-6
    # An actual CPU gradient update through RLinf's loss entrypoint.
    _, rollout = policy.predict_action_batch({"states": torch.randn(2, 780), "critic_states": torch.randn(2, 1390)})
    output = policy.default_forward(rollout["forward_inputs"])
    args.update(logprobs=output["logprobs"], old_logprobs=rollout["prev_logprobs"],
                values=output["values"], prev_values=rollout["prev_values"], returns=torch.ones(2, 1))
    loss, _ = policy_loss(**args)
    loss = loss - .005 * output["entropy"].sum(-1).mean()
    before = policy.actor_mean.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    optimizer.step()
    assert not torch.equal(before, policy.actor_mean.weight)
    assert policy.actor_norm.count.item() == 0
    # The production worker calls this method after PPO on its real FSDP root.
    import tempfile
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    with tempfile.TemporaryDirectory() as directory:
        torch.distributed.init_process_group("gloo", init_method=f"file://{directory}/pg", rank=0, world_size=1)
        try:
            fsdp = FullyShardedDataParallel(policy, device_id=torch.device("cpu"), sharding_strategy=ShardingStrategy.NO_SHARD)
            assert hasattr(fsdp, "update_normalizers")
            fsdp.update_normalizers(rollout["forward_inputs"]["raw_states"], rollout["forward_inputs"]["raw_critic_states"])
            assert fsdp.actor_norm.count.item() == 2
            state = fsdp.state_dict()
            restored = RPOFlatPolicy(780, 23, 1, True, False, critic_obs_dim=1390)
            restored.load_state_dict(state, strict=True)
            assert restored.actor_norm.count.item() == 2
        finally:
            torch.distributed.destroy_process_group()
    print("Patched RLinf factory, scoped clipped MSE, joint likelihood ratio and CPU gradient check passed")
    print("Actual FSDP root normalizer method access and full state-dict roundtrip passed")


if __name__ == "__main__":
    main()
    import sys
    if "--patched" in sys.argv:
        patched_rlinf_check()
