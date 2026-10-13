import pytest

RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7"


def test_reachable_recipe_protects_control_optimizer_and_short_budget():
    from eb.domain.rlinf_form import build_overrides
    from eb.domain.training_recipe import RECIPES, recipe_timeout_h, validate_update_target
    from eb.capacity.estimator import estimate_job
    from eb.domain.job import JobSpec

    recipe = RECIPES[RECIPE]
    assert recipe.embodiment_env_id == "ee_rpo_v2_bfm_23dof_v2"
    values = build_overrides(RECIPE, 4, "reachable", [])
    for item in ("env.train.total_num_envs=6144", "actor.model.batch_size=2048",
                 "+actor.model.optimizer_profile=stabilized_v7",
                 "+actor.model.control_profile=reachable_v2",
                 "+actor.model.auxiliary_profile=soft_velocity_v7"):
        assert item in values
    validate_update_target(524, start=24, recipe_id=RECIPE)
    with pytest.raises(ValueError):
        validate_update_target(525, start=24, recipe_id=RECIPE)
    assert recipe_timeout_h(RECIPE, "training") == 2
    estimate = estimate_job(JobSpec(kind="train.rl", embodiment_env_id=recipe.embodiment_env_id, task_id=recipe.task_id,
                                   params={"template_id": RECIPE}, resources={"gpus": 4}))
    assert estimate.basis["inputs"]["native_trainable_parameters"] == 435116484
    assert estimate.basis["inputs"]["num_envs"] == 6144


def test_optimizer_clips_before_adam_and_keeps_raw_and_applied_norms():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from eb.runners.bfm_ufo_agent import OPTIMIZERS, instrument_updates

    agent = SimpleNamespace()
    params = {}
    for name in OPTIMIZERS:
        param = torch.nn.Parameter(torch.zeros(2))
        setattr(agent, name, torch.optim.Adam([param], lr=.001))
        params[name] = param
    metrics = instrument_updates(agent, optimizer_profile="stabilized_v7")
    for name, param in params.items():
        param.grad = torch.tensor([60., 80.])
        getattr(agent, name).step()
        limit = 1. if name == "actor_optimizer" else 5.
        assert metrics[name + "/grad_norm_l2"] == pytest.approx(100.)
        assert metrics[name + "/grad_norm_after_clip_l2"] <= limit + 1e-5
        assert float(param.grad.norm()) <= limit + 1e-5
        assert metrics[name + "/parameter_delta_l2"] > 0
    params["actor_optimizer"].grad = torch.tensor([float("nan"), 0.])
    with pytest.raises(RuntimeError):
        agent.actor_optimizer.step()


def test_stabilized_native_config_preserves_network_and_changes_declared_fields():
    pytest.importorskip("torch")
    from eb.runners.bfm_ufo_agent import agent_config

    common = {"architecture_profile": "native_fb_v1", "hidden_dim": 2048, "hidden_layers": 6,
              "batch_size": 2048, "rollout_expert_trajectories": False}
    old = agent_config({**common, "auxiliary_profile": "soft_velocity_v5"}, device="cpu").model_dump()
    new = agent_config({**common, "auxiliary_profile": "soft_velocity_v7",
                        "optimizer_profile": "stabilized_v7"}, device="cpu").model_dump()
    assert new["model"]["archi"] == old["model"]["archi"]
    assert new["aux_rewards"] == old["aux_rewards"]
    assert new["aux_rewards_scaling"]["limits_dof_vel"] == -25.
    assert new["train"]["lr_discriminator"] == 3e-6
    assert new["train"]["clip_grad_norm"] == 0  # all six are clipped once at the Adam boundary


def test_reachable_wrapper_preserves_physical_action_trace(monkeypatch):
    pytest.importorskip("torch")
    import sys
    from types import SimpleNamespace
    from eb.runners.rpo_ufo_isaac import RPOUFOIsaacEnv

    class Env:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setitem(sys.modules, "rlinf.envs.sim.isaaclab.isaaclab_env", SimpleNamespace(IsaaclabBaseEnv=Env))
    env = RPOUFOIsaacEnv(SimpleNamespace(init_params=SimpleNamespace(id="RPO-UFO-Isaac-RSI-reachable-v7")))
    keys = ("states", "privileged_state", "last_action", "history_actor", "episode_id", "step_id",
            "ufo_aux_rewards", "ufo_action_trace")
    obs = {key: object() for key in keys}
    assert env._wrap_obs(obs)["ufo_action_trace"] is obs["ufo_action_trace"]


def test_reachable_resolved_config_rejects_mixed_control_and_optimizer(monkeypatch):
    pytest.importorskip("torch")
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from eb.domain.rlinf_form import get_template, build_overrides
    from eb.rpo_bfm import CODEC_V2_HASH
    from eb.runners.bfm_ufo_worker import validate_ufo_runtime_config

    monkeypatch.setenv("EMBODIED_PATH", "/workspace/rlinf/examples/embodiment")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "eb/runners/config"), version_base=None):
        cfg = compose(config_name=get_template(RECIPE).config_name,
            overrides=build_overrides(RECIPE, 4, "fixture", []) + [f"+actor.model.codec_sha256={CODEC_V2_HASH}"])
    validate_ufo_runtime_config(cfg)
    cfg.actor.model.control_profile = "v1"
    with pytest.raises(ValueError, match="approved"):
        validate_ufo_runtime_config(cfg)
