import pytest

from eb.robotwin import (
    ROBOTWIN_EMBODIMENTS,
    ROBOTWIN_HF_REVISION,
    ROBOTWIN_MODELS,
    ROBOTWIN_TASKS,
    get_download_profile,
    raw_archive_profile,
)


def test_catalog_has_five_native_embodiments_50_tasks_and_six_models():
    assert set(ROBOTWIN_EMBODIMENTS) == {
        "aloha-agilex", "piper", "franka-panda", "ARX-X5", "ur5-wsg"
    }
    assert len(ROBOTWIN_TASKS) == 50
    assert len(set(ROBOTWIN_TASKS)) == 50
    assert set(ROBOTWIN_MODELS) == {
        "lingbot-vla-2", "pi05", "gr00t-n1.7", "galaxea-g0", "dm05", "xiaomi-robotics-1"
    }
    assert len(ROBOTWIN_HF_REVISION) == 40


def test_raw_archive_profile_binds_task_scenario_and_native_embodiment():
    profile = raw_archive_profile("adjust_bottle", "randomized", "piper")
    assert profile.repo_id == "TianxingChen/RoboTwin2.0"
    assert profile.allow_patterns == ("dataset/adjust_bottle/piper_randomized_500.zip",)
    assert profile.native_embodiment == "piper"
    assert profile.scenario == "randomized"
    assert profile.embodiment_env_id == "ee_robotwin_piper_native_v1"


@pytest.mark.parametrize("task,scenario,embodiment", [
    ("../escape", "clean", "piper"),
    ("adjust_bottle", "other", "piper"),
    ("adjust_bottle", "clean", "unknown"),
])
def test_raw_archive_profile_rejects_unregistered_selection(task, scenario, embodiment):
    with pytest.raises(ValueError):
        raw_archive_profile(task, scenario, embodiment)


def test_lerobot_profile_is_pinned_and_requires_aloha_probe():
    profile = get_download_profile("aloha_lerobot_v21")
    assert profile.revision == ROBOTWIN_HF_REVISION
    assert profile.allow_patterns == ("lerobot_dataset/RoboTwin_lerobot_v21.zip",)
    assert profile.native_embodiment == "aloha-agilex"
    assert profile.requires_probe is True


def test_model_catalog_keeps_readiness_scoped_to_verified_embodiments():
    assert ROBOTWIN_MODELS["pi05"].ready is True
    assert ROBOTWIN_MODELS["lingbot-vla-2"].ready is True
    assert ROBOTWIN_MODELS["dm05"].ready is True
    assert ROBOTWIN_MODELS["pi05"].direct_embodiments == ("aloha-agilex",)
    assert ROBOTWIN_MODELS["gr00t-n1.7"].cross_embodiment is True
    assert ROBOTWIN_MODELS["gr00t-n1.7"].ready is True
    assert ROBOTWIN_MODELS["gr00t-n1.7"].direct_embodiments == ("aloha-agilex",)
    assert ROBOTWIN_MODELS["xiaomi-robotics-1"].ready is True
    assert ROBOTWIN_MODELS["xiaomi-robotics-1"].direct_embodiments == ("aloha-agilex",)
    assert ROBOTWIN_MODELS["galaxea-g0"].cross_embodiment == "limited"
    assert ROBOTWIN_MODELS["galaxea-g0"].ready is False
