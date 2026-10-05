from types import SimpleNamespace

from eb.api.routes_eval import robotwin_candidate


def test_robotwin_candidate_requires_pinned_package_and_eval_readiness():
    suite = SimpleNamespace(embodiment_env_id="ee_robotwin_aloha_joint14_v1")
    model = SimpleNamespace(id="mv_1", name="pi05", family_id="mf_pi05",
        embodiment_env_id=suite.embodiment_env_id, io_signature={},
        card={"readiness":"adapter_ready", "eval_template_id":"robotwin_pi05_eval"})
    artifact = SimpleNamespace(sha256="sha256:x", kind="model_package", pinned=True)
    assert robotwin_candidate(model, artifact, suite)["readiness"] == "adapter_ready"
    artifact.pinned = False
    result = robotwin_candidate(model, artifact, suite)
    assert result["readiness"] == "blocked"
    assert result["blockers"][0]["code"] == "model_package_incomplete"


def test_robotwin_candidate_remains_runnable_after_smoke():
    suite = SimpleNamespace(embodiment_env_id="ee_robotwin_aloha_joint14_v1")
    model = SimpleNamespace(id="mv_1", name="pi05", family_id="mf_pi05",
        embodiment_env_id=suite.embodiment_env_id, io_signature={},
        card={"readiness":"smoke_passed", "eval_template_id":"robotwin_pi05_eval"})
    artifact = SimpleNamespace(sha256="sha256:x", kind="model_package", pinned=True)
    result = robotwin_candidate(model, artifact, suite)
    assert result["readiness"] == "smoke_passed"
    assert result["blockers"] == []
