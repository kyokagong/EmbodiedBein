from eb.adapters import equivalent_command, get_adapter
from eb.domain.job import JobKind, JobSpec


def test_isaac_adapter_snapshot():
    spec = JobSpec(
        kind=JobKind.TRAIN_RL,
        contract_id="ec_bhl_biped_12dof_v1",
        name="biped-vel-s0",
        params={
            "task": "Velocity-Berkeley-Humanoid-Lite-Biped-v0",
            "num_envs": 4096,
            "max_iterations": 6000,
            "seed": 0,
        },
        overrides=["agent.algorithm.entropy_coef=0.006"],
        image="eb/isaac-lab:2.3.2",
    )
    rendered = get_adapter(spec).render(spec, "run_test")
    assert rendered.command[:3] == ["/isaac-sim/python.sh", "scripts/rsl_rl/train.py", "--task"]
    assert "Velocity-Berkeley-Humanoid-Lite-Biped-v0" in rendered.command
    assert "--num_envs" in rendered.command
    assert "4096" in rendered.command
    assert "logs" in rendered.writable_paths
    pythonpath = rendered.env.get("PYTHONPATH", "")
    for pkg in (
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite",
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite_assets",
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite_lowlevel",
    ):
        assert pkg in pythonpath.split(":")
    cmd = equivalent_command(spec, "run_test")
    assert "docker run" in cmd
    assert "--headless" in cmd


def test_export_policy_uses_play_export_only():
    spec = JobSpec(
        kind=JobKind.EXPORT_POLICY,
        contract_id="ec_bhl_biped_12dof_v1",
        name="biped-export",
        params={"task": "Velocity-Berkeley-Humanoid-Lite-Biped-v0", "num_envs": 16, "seed": 0},
        overrides=["--load_run", "2026-08-20_biped-smoke", "--checkpoint", "model_4.pt"],
        image="eb/isaac-lab:2.3.2",
    )
    rendered = get_adapter(spec).render(spec, "run_export")
    assert rendered.command[:3] == ["/isaac-sim/python.sh", "scripts/rsl_rl/play.py", "--task"]
    assert "--export_only" in rendered.command
    assert "--load_run" in rendered.command
    assert "model_4.pt" in rendered.command
    assert "--max_iterations" not in rendered.command
    assert "berkeley_humanoid_lite_assets" in rendered.env["PYTHONPATH"]
