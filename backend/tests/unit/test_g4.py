from pathlib import Path

from eb.domain.contract import load_env
from eb.gates.engine import g4_export

ROOT = Path(__file__).resolve().parents[3]


def test_g4_rejects_bad_yaml():
    ctr = load_env(ROOT / "contracts/ee_bhl_biped_12dof_v1.yaml")
    bad = {"num_observations": 1, "num_actions": 1}
    r = g4_export(bad, ctr)
    assert not r.passed
