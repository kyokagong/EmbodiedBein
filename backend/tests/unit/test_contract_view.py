from pathlib import Path

from eb.domain.contract import load_env
from eb.domain.contract_view import (
    action_terms_view,
    assertion_table,
    frequency_view,
    joint_groups_view,
    observation_layout_view,
)

ROOT = Path(__file__).resolve().parents[3]


def test_humanoid_layout_ranges_cover_75():
    ctr = load_env(ROOT / "contracts/ee_bhl_humanoid_22dof_v1.yaml")
    layout = observation_layout_view(ctr)
    assert [t["term"] for t in layout] == [
        "velocity_commands",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    assert layout[0]["index_start"] == 0 and layout[0]["index_end"] == 2
    assert layout[-1]["index_start"] == 53 and layout[-1]["index_end"] == 74
    assert layout[-1]["index_end"] + 1 == ctr.observation.dim


def test_wbc_inserts_arm_reference_and_dual_action():
    ctr = load_env(ROOT / "contracts/ee_bhl_wbc_22dof_v1.yaml")
    layout = observation_layout_view(ctr)
    assert layout[1]["term"] == "arm_reference"
    assert layout[1]["dim"] == 10
    assert layout[1]["index_start"] == 3
    terms = action_terms_view(ctr)
    assert [t["name"] for t in terms] == ["arm", "leg"]
    assert terms[0]["dim"] == 10 and terms[1]["dim"] == 12
    assert sum(t["dim"] for t in terms) == 22


def test_fixedhand_does_not_concat_camera():
    ctr = load_env(ROOT / "contracts/ee_bhl_fixedhand_10dof_v1.yaml")
    layout = observation_layout_view(ctr)
    assert ctr.observation.concat is False
    assert layout == [
        {
            "term": "state",
            "dim": 10,
            "source": "encoder",
            "index_start": 0,
            "index_end": 9,
            "color": layout[0]["color"],
        }
    ]
    assert ctr.observation.modalities[0].name == "camera_top"
    assert ctr.observation.modalities[0].shape == [224, 224, 3]


def test_biped_joint_tree_is_legs_only():
    ctr = load_env(ROOT / "contracts/ee_bhl_biped_12dof_v1.yaml")
    groups = joint_groups_view(ctr)
    assert [g["name"] for g in groups] == ["leg"]
    assert groups[0]["count"] == 12
    assert groups[0]["joints"][0] == "leg_left_hip_roll_joint"
    assert len(groups[0]["joints"]) == 12


def test_assertion_table_always_has_four_rows_even_when_passing():
    ctr = load_env(ROOT / "contracts/ee_bhl_humanoid_22dof_v1.yaml")
    rows = assertion_table(ctr)
    assert len(rows) == 4
    assert all(r["passed"] for r in rows)
    freq = frequency_view(ctr)
    assert freq["derived_hz"] == 25
    assert freq["ok"] is True
    wbc = load_env(ROOT / "contracts/ee_bhl_wbc_22dof_v1.yaml")
    assert frequency_view(wbc)["derived_hz"] == 50
