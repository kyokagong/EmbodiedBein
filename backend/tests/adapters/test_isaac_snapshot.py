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
    cmd = equivalent_command(spec, "run_test")
    assert "docker run" in cmd
    assert "--headless" in cmd
