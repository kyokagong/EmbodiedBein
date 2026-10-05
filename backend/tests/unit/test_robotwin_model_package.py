from types import SimpleNamespace

import pytest

from eb.orchestrator.collect import robotwin_model_package


def test_robotwin_model_package_is_immutable_and_maps_eval_recipe():
    job = SimpleNamespace(id="job_1", spec={"name":"pi05 smoke", "embodiment_env_id":"ee_robotwin_aloha_joint14_v1",
        "inputs":{"base_model":"mdl_parent"}, "params":{"template_id":"robotwin_pi05_sft", "model_name":"我的 π0.5 模型"}})
    run = SimpleNamespace(id="run_1")
    checkpoint = {"path":"/data/kk/models/runs/run_1", "bytes":123, "step":2,
        "manifest":{"schema_version":"eb/checkpoint/v1", "files":[
            {"path":"model/model_state_dict/full_weights.pt", "bytes":7, "sha256":"a" * 64},
            {"path":"model/physical-intelligence/robotwin/norm_stats.json", "bytes":4, "sha256":"b" * 64}]}}
    package = robotwin_model_package(job, run, checkpoint)
    assert package["artifact"]["sha256"].startswith("sha256:")
    assert package["artifact"]["storage"] == checkpoint["path"]
    assert package["model"]["card"]["eval_template_id"] == "robotwin_pi05_eval"
    assert package["model"]["card"]["readiness"] == "adapter_ready"
    assert package["model"]["name"] == "我的 π0.5 模型"
    assert package["model"]["io_signature"]["action_profile"] == "aloha_joint14_position_v1"


def test_robotwin_model_package_keeps_unverified_bridge_blocked():
    job = SimpleNamespace(id="job_1", spec={"name":"xr1 smoke", "embodiment_env_id":"ee_robotwin_aloha_joint14_v1",
        "inputs":{}, "params":{"template_id":"robotwin_xr1_sft"}})
    package = robotwin_model_package(job, SimpleNamespace(id="run_1"), {
        "path":"/data/kk/models/runs/run_1", "bytes":1, "step":1,
        "manifest":{"files":[{"path":"model/model_state_dict/full_weights.pt", "bytes":1, "sha256":"c" * 64}]}})
    assert package["model"]["card"]["readiness"] == "blocked"
    assert package["model"]["card"]["blocker_code"] == "controller_bridge_unverified"


def test_robotwin_pi05_package_requires_robotwin_normalization_assets():
    job = SimpleNamespace(id="job_1", spec={"name":"pi05", "embodiment_env_id":"ee_robotwin_aloha_joint14_v1",
        "inputs":{}, "params":{"template_id":"robotwin_pi05_sft"}})
    checkpoint = {"path":"/data/kk/models/runs/run_1", "bytes":1, "step":1,
        "manifest":{"files":[{"path":"model/model_state_dict/full_weights.pt", "bytes":1,
                                "sha256":"c" * 64}]}}
    with pytest.raises(ValueError, match="normalization"):
        robotwin_model_package(job, SimpleNamespace(id="run_1"), checkpoint)
