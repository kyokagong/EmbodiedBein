from eb.adapters import get_adapter
from eb.domain.job import JobKind, JobSpec


def test_rlinf_wbc_forces_log_path_and_placement():
    spec = JobSpec(
        kind=JobKind.TRAIN_RLFT,
        embodiment_env_id="ee_bhl_wbc_22dof_v1",
        task_id="tsk_bhl_wbc_tracking",
        name="wbc-test",
        params={
            "template_id": "berkeley_wbc_ppo_mlp",
            "config_name": "berkeley_wbc_ppo_mlp",
            "config_path": "examples/embodiment/config",
            "script": "examples/embodiment/train_embodied_agent.py",
            "ray_port": 6401,
        },
        resources={"gpus": 2, "shm_gb": 32},
        code_refs={
            "embodiment": {"kind": "git_external", "path": "/data/kk/Berkeley-Humanoid-Lite"},
            "stack": {"kind": "git_external", "path": "/data/kk/RLinf"},
        },
        image="eb/rlinf:0.4.0-emb",
    )
    rendered = get_adapter(spec).render(spec, "trn_test")
    joined = " ".join(rendered.command)
    assert "train_embodied_agent.py" in joined
    assert "runner.logger.log_path=/workspace/output" in joined
    assert "actor,env,rollout=0,1" in joined
    assert "actor_rollout=2" not in joined
    assert any(m[1] == "/workspace/berkeley" and m[2] == "ro" for m in rendered.mounts)
    assert any(m[1] == "/workspace/output" and m[2] == "rw" for m in rendered.mounts)
    assert rendered.shm_gb == 32
