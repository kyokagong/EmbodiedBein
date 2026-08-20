from pathlib import Path

from eb.domain.contract import load_contract
from eb.domain.fsm import IllegalTransition, JobStatus, transition
from eb.domain.job import JobKind, JobSpec
from eb.gates.engine import GateContext, g1_submit, g5_robot

ROOT = Path(__file__).resolve().parents[3]


def test_humanoid_contract_self_consistent():
    ctr = load_contract(ROOT / "contracts/ec_bhl_humanoid_22dof_v1.yaml")
    assert ctr.dof == 22
    assert ctr.observation.dim == 75
    r = ctr.validate_self()
    assert r.passed, r.violations


def test_biped_and_wbc_and_hand():
    for name, dof, obs, hz in [
        ("ec_bhl_biped_12dof_v1.yaml", 12, 45, 25),
        ("ec_bhl_wbc_22dof_v1.yaml", 22, 85, 50),
        ("ec_bhl_fixedhand_10dof_v1.yaml", 10, 10, 20),
    ]:
        ctr = load_contract(ROOT / "contracts" / name)
        assert ctr.dof == dof
        assert ctr.observation.dim == obs
        assert ctr.control.policy_hz == hz
        assert ctr.validate_self().passed, ctr.validate_self().violations


def test_g1_rejects_task_mismatch():
    ctr = load_contract(ROOT / "contracts/ec_bhl_biped_12dof_v1.yaml")
    spec = JobSpec(
        kind=JobKind.TRAIN_RL,
        contract_id=ctr.id,
        params={"task": "Velocity-Berkeley-Humanoid-Lite-v0"},
    )
    r = g1_submit(spec, ctr, GateContext(dataset_versions=[]))
    assert not r.passed
    assert r.violations[0].path == "task"


def test_g5_blocks_expired_calibration():
    ctr = load_contract(ROOT / "contracts/ec_bhl_biped_12dof_v1.yaml")
    r = g5_robot(ctr, firmware_n_joints=12, calibration_expired=True, latency_ms=3)
    assert not r.passed


def test_fsm_illegal():
    try:
        transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)
        raise AssertionError("should fail")
    except IllegalTransition:
        pass
    assert transition(JobStatus.QUEUED, JobStatus.ADMITTED) == JobStatus.ADMITTED
