import pytest

torch = pytest.importorskip("torch")
from eb.runners.bfm_ufo_worker import UFOLearner
from eb.runners.bfm_ufo_agent import OPTIMIZERS


class Agent:
    def __init__(self):
        self.calls = []
        for name in OPTIMIZERS:
            setattr(self, name, type("Adam", (), {"state": {"p": {"step": torch.tensor(0.)}}})())

    def update(self, replay, step):
        self.calls.append(step)
        for name in OPTIMIZERS:
            getattr(self, name).state["p"]["step"] += 1
        return {"loss": torch.tensor(.4)}


class Replay:
    received = 0
    def append(self, transitions):
        count = transitions["count"]
        self.received += count
        return count


def test_reset_projection_metrics_distinguish_measured_pose_and_target_anchor():
    from eb.rpo_bfm import ACTION_GAIN
    from eb.runners.bfm_ufo_worker import reset_control_diagnostics
    offset = torch.zeros(3, 23)
    offset[0, 0] = .75
    offset[2, 0] = 10
    last = (offset / (.25 * torch.tensor(ACTION_GAIN))).clamp(-1, 1)
    values = {"step_id": torch.tensor([[0], [0], [1]]),
        "observation": {"state": torch.cat((offset, torch.zeros(3, 29)), -1), "last_action": last}}
    result = reset_control_diagnostics(values)
    assert result["reset_control_samples"] == 2
    assert result["reset_projection_residual_max_rad"] == .5
    assert result["reset_projection_outside_fraction"] == .5


def test_warmup_has_zero_joint_updates():
    agent = Agent()
    learner = UFOLearner(agent, Replay(), object(), target=24)
    assert learner.train_round({"count": 10240})["learner_joint_updates"] == 0
    assert not agent.calls
    assert learner.train_round({"count": 1024})["learner_joint_updates"] == 16
    assert agent.calls == [11264] * 16


def test_initialization_baseline_never_updates_the_agent():
    agent = Agent()
    learner = UFOLearner(agent, Replay(), object(), target=0)
    assert learner.train_round({"count": 64})["learner_joint_updates"] == 0
    assert not agent.calls


def test_rollout_diagnostics_include_warmup_terminals_and_action_saturation():
    learner = UFOLearner(Agent(), Replay(), object(), target=0)
    metrics = learner.train_round({"count": 2, "action": torch.tensor([[1., .1], [-1., .2]]),
        "next": {"terminated": torch.tensor([[True], [False]]), "truncated": torch.tensor([[False], [True]])}})
    assert metrics["action_saturation_fraction"] == .5
    assert metrics["terminated_transitions"] == 1
    assert metrics["truncated_transitions"] == 1
    assert metrics["replay_draw_new_ratio"] == 0


def test_joint_boundary_stops_without_overshoot():
    agent = Agent()
    learner = UFOLearner(agent, Replay(), object(), target=24)
    learner.train_round({"count": 11264})
    metrics = learner.train_round({"count": 1024})
    assert metrics["learner_joint_updates"] == 24
    assert len(agent.calls) == 24
    assert learner.train_round({"count": 1024})["learner_joint_updates"] == 24
    assert len(agent.calls) == 24


def test_every_native_update_has_a_raw_metric_point():
    class Writer:
        points = []
        def add_scalar(self, key, value, step):
            self.points.append((key, value, step))
        def flush(self):
            pass
    learner = UFOLearner(Agent(), Replay(), object(), target=24)
    learner.writer = Writer()
    learner.train_round({"count": 11264})
    learner.train_round({"count": 1024})
    assert [step for key, _, step in learner.writer.points if key == "train/ufo/loss"] == list(range(1, 25))


def test_six_optimizers_have_one_owner():
    from eb.runners.bfm_ufo_worker import UFOActorWorker
    from rlinf.scheduler import Worker
    assert UFOActorWorker.__bases__ == (Worker,)
    assert not hasattr(UFOActorWorker, "build_optimizer")


@pytest.mark.parametrize("recipe_id,envs,samplers", [("rpo_bfm_ufo_agent_rlinf_isaac_v1",16,1), ("rpo_bfm_ufo_agent_rlinf_isaac_4gpu_v1",192,3), ("rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",192,3), ("rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3",192,3), ("rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5",192,3), ("rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6",6144,3)])
def test_image_rlinf_validates_frozen_hydra_config_without_starting_cluster(monkeypatch,recipe_id,envs,samplers):
    from pathlib import Path
    from types import SimpleNamespace
    from hydra import compose, initialize_config_dir
    import rlinf.config as runtime
    from eb.domain.rlinf_form import get_template, build_overrides

    monkeypatch.setenv("EMBODIED_PATH", "/workspace/rlinf/examples/embodiment")
    monkeypatch.setattr(runtime, "Cluster", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "HybridComponentPlacement",
        lambda *args: SimpleNamespace(get_world_size=lambda name: 1 if name == "actor" else samplers))
    config_dir = Path(__file__).resolve().parents[2] / "eb/runners/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=get_template(recipe_id).config_name,
            overrides=build_overrides(recipe_id, 4 if samplers == 3 else 1, "fixture", []))
    checked = runtime.validate_cfg(cfg)
    assert checked.actor.training_backend == "ufo_agent"
    assert checked.actor.optim is None and checked.actor.fsdp_config is None
    assert checked.env.train.total_num_envs == envs
    cfg.actor.training_backend = "fsdp"
    with pytest.raises(ValueError, match="approved native recipe"):
        runtime.validate_cfg(cfg)


def test_safe_trace_survives_native_auto_reset_and_trajectory_builder():
    from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv
    from rlinf.data.schema.embodied_types import EnvOutput
    from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder

    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs = 2
    final = {"states": torch.full((2, 52), 7.), "ufo_action_trace": torch.full((2, 118), 3.)}
    final["ufo_action_trace"][1] = 5.
    reset = {key: value.clone() for key, value in final.items()}
    def reset_done(*, env_ids):
        assert env_ids.tolist() == [0]
        for value in reset.values():
            value[env_ids] = 0.
        return reset, {}
    env.reset = reset_done
    dones = torch.tensor([True, False])
    obs, infos = env._handle_auto_reset(dones, final, {})
    output = EnvOutput(obs=obs, final_obs=infos["final_observation"], dones=dones).to_dict()
    assert output["obs"]["ufo_action_trace"][0].sum() == 0
    # Same final-observation selection used by the native EnvWorker after auto-reset.
    next_obs = output["final_obs"] if output["dones"].any() else output["obs"]
    builder = EmbodiedTrajectoryBuilder()
    builder.append_transitions(final, {k: v for k, v in next_obs.items() if isinstance(v, torch.Tensor)})
    trajectory = builder.to_trajectory()
    assert (trajectory.next_obs["ufo_action_trace"][0, 0] == 3).all()
    assert (trajectory.next_obs["ufo_action_trace"][0, 1] == 5).all()
    assert (trajectory.next_obs["states"] == 7).all()


def test_rpc_receives_real_rlinf_trajectory_and_terminal_flags():
    import asyncio
    from types import SimpleNamespace
    from eb.runners.bfm_ufo_agent import OBS_DIMS
    from eb.runners.bfm_ufo_worker import UFOActorWorker, rollout_transitions
    from rlinf.data.schema.embodied_types import Trajectory

    current = {("states" if key == "state" else key): torch.zeros(4, 2, dim) for key, dim in OBS_DIMS.items()}
    terminal = {key: torch.full_like(value, 7.) for key, value in current.items()}
    terminal["ufo_aux_rewards"] = torch.ones(4, 2, 8)
    action = torch.full((4, 2, 1, 23), .2)
    flags = torch.zeros(5, 2, 1, dtype=torch.bool)
    flags[-1, 0] = True
    timeout = torch.zeros_like(flags)
    timeout[-1, 1] = True
    trajectory = Trajectory(actions=action, curr_obs=current, next_obs=terminal,
        forward_inputs={"action": action[:, :, 0], "z": torch.ones(4, 2, 256),
            **{key: torch.zeros(4, 2, 1, dtype=torch.long) for key in ("episode_id", "step_id", "policy_version")}},
        terminations=flags, truncations=timeout)
    class Pending:
        async def async_wait(self):
            return trajectory
    class Channel:
        def get(self, *, async_op):
            assert async_op
            return Pending()
    worker = object.__new__(UFOActorWorker)
    from eb.runners.bfm_ufo_agent import AUX_NAMES
    worker.aux_names = AUX_NAMES
    worker.placement = SimpleNamespace(get_world_size=lambda name: 1)
    worker.cfg = SimpleNamespace(rollout=SimpleNamespace(pipeline_stage_num=1))
    asyncio.run(worker.recv_rollout_trajectories(Channel()))
    assert worker.compute_advantages_and_returns() == {"ufo/received_transitions": 8}
    batch = worker.transitions
    assert batch["next"]["terminated"][-2:].tolist() == [[True], [False]]
    assert batch["next"]["truncated"][-2:].tolist() == [[False], [True]]
    assert (batch["next"]["observation"]["state"] == 7).all()
    worker.rollout_batch["forward_inputs"]["action"] *= 0
    with pytest.raises(ValueError, match="behavior"):
        rollout_transitions(worker.rollout_batch)
