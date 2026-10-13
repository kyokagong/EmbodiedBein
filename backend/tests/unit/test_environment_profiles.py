from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.api.routes_environments import (
    EnvironmentEdit,
    clone_environment,
    edit_environment,
    publish_environment,
)
from eb.api.routes_eval import _bind_environment
from eb.api.routes_train import TrainIn, _build_spec
from eb.domain.environment_profile import (
    compatible_profile,
    profile_hash,
    rpo_eval_arguments,
    validate_definition,
)
from eb.runners.rpo_wbc_artifact import case_success
from eb.runners.rpo_wbc_extension import training_settings_from_env
from eb.store.models import EmbodimentRow, EnvironmentProfile


@pytest.mark.asyncio
async def test_bfm_catalog_derives_compatible_profile_without_mutating_old_profile():
    from unittest.mock import Mock
    from eb.environment_catalog import seed_environment_profiles
    from eb.rpo_bfm import ENV_ID, TASK_ID, UFO_REPAIR_ACCEPTANCE_SUITE, frozen_eval_suite_spec
    suite = SimpleNamespace(id=UFO_REPAIR_ACCEPTANCE_SUITE, name="repair acceptance", frozen=True,
        embodiment_env_id=ENV_ID, spec=frozen_eval_suite_spec(UFO_REPAIR_ACCEPTANCE_SUITE, "acceptance"))
    old = SimpleNamespace(id=f"env_eval_{suite.id}", task_id="", status="deprecated", content_hash="old")
    scalars = AsyncMock(side_effect=[SimpleNamespace(all=lambda: [ENV_ID]), SimpleNamespace(all=lambda: [suite])])
    session = SimpleNamespace(scalars=scalars, get=AsyncMock(side_effect=lambda table, id_: old if id_ == old.id else None), add=Mock())
    await seed_environment_profiles(session)
    rows = [call.args[0] for call in session.add.call_args_list if call.args[0].usage == "eval"]
    assert len(rows) == 1 and rows[0].id != old.id
    compatible_profile(rows[0], "eval", suite.id, ENV_ID, TASK_ID)
    assert "0.20" in rows[0].definition["success"]
    assert old.task_id == "" and old.status == "deprecated" and old.content_hash == "old"


def test_environment_settings_are_executable_and_bounded():
    definition = {
        "task": "直立并直行",
        "reward": "速度跟踪",
        "failure": "跌倒",
        "success": "直行 4m",
        "settings": {
            "target_speed_mps": 0.3,
            "stand_fraction": 0.5,
            "velocity_reward_weight": 3.0,
            "upright_reward_weight": 0.5,
        },
    }
    assert (
        validate_definition("train", "rpo_v2_wbc_stand_walk_ppo", definition)["settings"]["target_speed_mps"]
        == 0.3
    )
    with pytest.raises(ValueError, match="不支持"):
        validate_definition("train", "libero_pi05_sft", {**definition, "settings": {"target_speed_mps": 0.3}})
    with pytest.raises(ValueError, match="范围"):
        validate_definition(
            "train", "rpo_v2_wbc_stand_walk_ppo", {**definition, "settings": {"stand_fraction": 2}}
        )


def test_profile_must_match_template_or_suite():
    row = type(
        "Profile",
        (),
        {
            "status": "active",
            "usage": "train",
            "template_id": "rpo_v2_wbc_stand_walk_ppo",
            "suite_id": None,
            "embodiment_env_id": "ee_rpo_v2_wbc_23dof_v1",
            "task_id": "tsk_rpo_stand_walk",
        },
    )()
    row.definition = {"task": "直行", "reward": "速度", "failure": "跌倒", "success": "评估", "settings": {}}
    row.content_hash = profile_hash(row)
    compatible_profile(
        row, "train", "rpo_v2_wbc_stand_walk_ppo", "ee_rpo_v2_wbc_23dof_v1", "tsk_rpo_stand_walk"
    )
    with pytest.raises(ValueError, match="不匹配"):
        compatible_profile(row, "train", "rpo_v2_wbc_ppo", "ee_rpo_v2_wbc_23dof_v1", "tsk_rpo_stand_walk")
    row.definition["reward"] = "悄悄改变"
    with pytest.raises(ValueError, match="哈希"):
        compatible_profile(
            row, "train", "rpo_v2_wbc_stand_walk_ppo", "ee_rpo_v2_wbc_23dof_v1", "tsk_rpo_stand_walk"
        )


@pytest.mark.asyncio
async def test_selected_training_environment_is_frozen_into_spec():
    definition = {
        "task": "直行",
        "reward": "速度",
        "failure": "跌倒",
        "success": "评估通过",
        "settings": {"target_speed_mps": 0.4},
    }
    profile = SimpleNamespace(
        id="env_revision",
        status="active",
        usage="train",
        template_id="rpo_v2_wbc_stand_walk_ppo",
        suite_id=None,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_stand_walk",
        definition=definition,
        content_hash="",
    )
    profile.content_hash = profile_hash(profile)
    session = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda table, key: (
                profile
                if table is EnvironmentProfile
                else SimpleNamespace(id="emb_roboparty_rpo")
                if table is EmbodimentRow
                else None
            )
        )
    )
    body = TrainIn(
        name="rpo trial",
        model_name="trial",
        template_id="rpo_v2_wbc_stand_walk_ppo",
        embodiment_id="emb_roboparty_rpo",
        environment_id="env_revision",
    )
    spec = await _build_spec(body, session)
    assert spec.params["environment_profile"]["hash"] == profile.content_hash
    assert spec.params["rpo_train_settings"]["target_speed_mps"] == 0.4


def test_runner_reads_selected_training_values(monkeypatch):
    monkeypatch.setenv("EB_RPO_TARGET_SPEED_MPS", "0.4")
    assert training_settings_from_env()["target_speed_mps"] == 0.4


def test_diagnostic_eval_threshold_changes_case_result():
    assert case_success(
        "stand_walk",
        1000,
        False,
        False,
        False,
        0,
        0.1,
        scenario="walk",
        distance_m=3.8,
        settings={"walk_min_forward_m": 3.5},
    )


def test_eval_settings_become_runner_arguments():
    assert rpo_eval_arguments(
        {"criterion": "stand_walk", "environment_settings": {"walk_min_forward_m": 3.5}}
    ) == ["--walk-min-forward-m", "3.5"]


@pytest.mark.asyncio
async def test_eval_environment_changes_plan_hash_and_is_diagnostic():
    suite = SimpleNamespace(
        id="suite_rpo_v2_wbc_isaac_stand_walk_v2",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        spec={"stand_max_drift_m": 0.5, "walk_min_forward_m": 4.0},
    )
    row = SimpleNamespace(
        id="env_revision",
        status="active",
        usage="eval",
        template_id=None,
        suite_id=suite.id,
        embodiment_env_id=suite.embodiment_env_id,
        task_id="tsk_rpo_stand_walk",
        content_hash="",
        definition={
            "task": "直行",
            "reward": "无",
            "failure": "跌倒",
            "success": "前进",
            "settings": {"stand_max_drift_m": 0.5, "walk_min_forward_m": 3.5},
        },
    )
    row.content_hash = profile_hash(row)
    session = SimpleNamespace(get=AsyncMock(return_value=row))
    plan = {"criterion": "stand_walk", "hash": "sha256:old"}
    snapshot = await _bind_environment(row.id, suite, row.task_id, plan, session)
    assert snapshot["hash"] == row.content_hash
    assert plan["hash"] != "sha256:old" and plan["diagnostic_environment"] is True
    with pytest.raises(ValueError, match="不匹配"):
        await _bind_environment(row.id, suite, "tsk_wrong", {}, session)


@pytest.mark.asyncio
async def test_edit_creates_a_new_version_without_mutating_published_environment():
    definition = {
        "task": "直行",
        "reward": "速度跟踪",
        "failure": "跌倒",
        "success": "独立评估",
        "settings": {},
    }
    source = EnvironmentProfile(
        id="env_source",
        name="原环境",
        usage="train",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_stand_walk",
        template_id="rpo_v2_wbc_stand_walk_ppo",
        suite_id=None,
        definition=definition,
        content_hash="sha256:source",
        status="active",
    )
    rows = {source.id: source}

    class Session:
        async def get(self, table, key):
            return rows.get(key) if table is EnvironmentProfile else None

        def add(self, row):
            rows[row.id] = row

        async def commit(self):
            pass

    session = Session()
    principal = SimpleNamespace(actor_type="user")
    clone = await clone_environment(source.id, session, principal)
    changed = {"velocity_reward_weight": 4.0}
    await edit_environment(clone["id"], EnvironmentEdit(name="修订", settings=changed), session, principal)
    published = await publish_environment(clone["id"], session, principal)
    assert published["status"] == "active" and published["content_hash"] != source.content_hash
    assert source.definition["settings"] == {} and source.name == "原环境"


@pytest.mark.asyncio
async def test_changing_training_task_rebinds_compatible_template():
    row = EnvironmentProfile(
        id="env_draft",
        name="草稿",
        usage="train",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_stand_walk",
        template_id="rpo_v2_wbc_stand_walk_ppo",
        suite_id=None,
        definition={"task": "直行", "reward": "速度", "failure": "跌倒", "success": "评估", "settings": {}},
        content_hash="sha256:old",
        status="draft",
    )
    session = SimpleNamespace(get=AsyncMock(return_value=row), commit=AsyncMock())
    changed = EnvironmentEdit(name="行走跟踪", template_id="rpo_v2_wbc_ppo")
    result = await edit_environment(row.id, changed, session, SimpleNamespace(actor_type="user"))
    assert result["template_id"] == "rpo_v2_wbc_ppo"
    assert result["task_id"] == "tsk_rpo_walk_upperbody_tracking"


@pytest.mark.asyncio
async def test_reward_description_follows_executable_weight():
    row = EnvironmentProfile(
        id="env_draft",
        name="草稿",
        usage="train",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_stand_walk",
        template_id="rpo_v2_wbc_stand_walk_ppo",
        suite_id=None,
        definition={
            "task": "旧任务",
            "reward": "旧权重",
            "failure": "跌倒",
            "success": "评估",
            "settings": {},
        },
        content_hash="sha256:old",
        status="draft",
    )
    session = SimpleNamespace(get=AsyncMock(return_value=row), commit=AsyncMock())
    result = await edit_environment(
        row.id,
        EnvironmentEdit(name="草稿", settings={"velocity_reward_weight": 4.0}),
        session,
        SimpleNamespace(actor_type="user"),
    )
    assert result["definition"]["settings"]["velocity_reward_weight"] == 4.0
    assert "速度跟踪 +4" in result["definition"]["reward"]
