import pytest

from eb.robotwin_suite import (
    asset_manifest_sha256,
    build_robotwin_full_suite_spec,
    build_robotwin_suite_spec,
)


def test_robotwin_suite_freezes_every_seed_and_manifest():
    spec = build_robotwin_suite_spec([100100000, 100100002], "sha256:" + "a" * 64)
    assert [case["seed"] for case in spec["cases"]] == [100100000, 100100002]
    assert spec["case_manifest_sha256"].startswith("sha256:")
    assert spec["action_chunk"]["consume_rule"] == "stop_on_success_timeout_or_termination"
    with pytest.raises(ValueError):
        build_robotwin_suite_spec([1, 1], "sha256:" + "a" * 64)


def test_asset_manifest_digest_covers_every_frozen_field():
    manifest = {"schema_version":"eb/robotwin-assets/v1", "status":"verified",
                "source":{"repo":"official", "revision":"abc"}}
    digest = asset_manifest_sha256(manifest)
    assert digest.startswith("sha256:")
    assert asset_manifest_sha256({**manifest, "sha256":digest}) == digest
    assert asset_manifest_sha256({**manifest, "source":{"repo":"official", "revision":"def"}}) != digest


def test_full_suite_freezes_all_tasks_cases_limits_and_round_robin_smoke_order():
    seeds = {"task_a":{"success_seeds":[1, 2]}, "task_b":{"success_seeds":[3]},
             "task_c":{"success_seeds":[4, 5]}}
    spec = build_robotwin_full_suite_spec(
        seeds, {"task_a":400, "task_b":500, "task_c":600},
        "sha256:" + "b" * 64, "sha256:" + "a" * 64)
    assert spec["suite"] == "robotwin_rlinf_full_clean_v1"
    assert len(spec["tasks"]) == 3 and len(spec["cases"]) == 5
    assert [case["task_id"] for case in spec["cases"][:3]] == ["task_a", "task_b", "task_c"]
    assert spec["tasks"][1]["max_episode_steps"] == 500
    assert spec["max_episode_steps"] == 600
    assert len({task["task_config_sha256"] for task in spec["tasks"]}) == 3
    with pytest.raises(ValueError):
        build_robotwin_full_suite_spec(seeds, {"task_a":400},
            "sha256:" + "b" * 64, "sha256:" + "a" * 64)
