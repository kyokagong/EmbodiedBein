from pathlib import Path

from eb.domain.contract import load_env


def test_rpo_pbfm_depth_contract_is_distinct_and_not_implicitly_published():
    path = Path(__file__).resolve().parents[3] / "contracts/ee_rpo_v2_pbfm_depth_23dof_v1.yaml"
    env = load_env(path)
    assert env.self_assertions() == []
    assert env.status == "draft"
    assert env.dof == env.action.dim == 23
    assert env.observation.dim == 380
    assert env.observation.concat is False
    assert env.observation.history_preflattened
    assert [(m.name, m.shape, m.dtype) for m in env.observation.modalities] == [
        ("depth_image", [8, 18, 32], "uint8")
    ]
    assert env.control.policy_hz == 50 and env.control.control_hz == 400
    assert env.control.decimation == 8 and env.control.sim_dt == 0.0025
    assert env.gains["base_ang_vel_scale"] == 0.25
    assert env.gains["last_action_stage"] == "normalized_clipped_before_motor_delay"
    assert env.gains["velocity_estimator_input_dim"] == 631
    assert env.gains["depth_sampled_ages"] == [35, 30, 25, 20, 15, 10, 5, 0]
    assert env.limits["hardware_verified"] is False


def test_g1_allows_only_the_exact_pbfm_draft_runtime_diagnostic():
    from eb.domain.job import JobKind, JobSpec
    from eb.gates.engine import GateContext, g1_submit

    root = Path(__file__).resolve().parents[3]
    env = load_env(root / "contracts/ee_rpo_v2_pbfm_depth_23dof_v1.yaml")
    probe = JobSpec(kind=JobKind.ASSET_VALIDATE, embodiment_env_id=env.id,
                    params={"operation": "probe_pbfm_runtime"})
    context = GateContext(dataset_versions=[])
    assert g1_submit(probe, env, context).passed
    for spec, contract in (
        (probe.model_copy(update={"kind": JobKind.TRAIN_RL}), env),
        (probe.model_copy(update={"params": {"operation": "unknown_probe"}}), env),
        (probe, env.model_copy(update={"status": "deprecated"})),
        (probe, env.model_copy(update={"id": "ee_other_draft_v1"})),
        (probe.model_copy(update={"embodiment_env_id": "ee_other_draft_v1"}), env),
    ):
        result = g1_submit(spec, contract, context)
        assert any(v.path == "embodiment_env.status" for v in result.violations)
    forbidden = probe.model_copy(update={"overrides": ["action.dim=1"]})
    result = g1_submit(forbidden, env, context)
    assert any(v.path == "overrides" for v in result.violations)
