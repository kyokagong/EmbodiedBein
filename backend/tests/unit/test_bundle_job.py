from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import json

import pytest


def test_bundle_job_renders_python_worker_with_only_declared_mounts(tmp_path, monkeypatch):
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = JobSpec(kind="bundle.build", embodiment_env_id="env", params={"bundle_plan": {
        "onnx_path": "/data/kk/embodiedbein/cas/aa/" + "a" * 64,
        "onnx_sha256": "a" * 64, "onnx_bytes": 4, "hash": "frozen"}}, runtime={"network":"none"})
    rendered = ToolsAdapter().render(spec, "run_test")
    assert rendered.command[:3] == ["python", "-m", "eb.bundle.worker"]
    assert any(mount[2] == "ro" for mount in rendered.mounts)
    assert not any("sock" in mount[0] for mount in rendered.mounts)


def test_worker_fails_when_numerical_check_is_unavailable(tmp_path, monkeypatch):
    from eb.bundle.worker import stage_bundle
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"fake")
    monkeypatch.setattr("eb.bundle.worker.read_signature", lambda _: {"obs": 2, "act": 1})
    monkeypatch.setattr("eb.bundle.worker.numerical_check", lambda *_: {"passed": False, "error":"missing runtime"})
    with pytest.raises(ValueError, match="numerical"):
        stage_bundle({"onnx_sha256": __import__('hashlib').sha256(b"fake").hexdigest(), "onnx_bytes": 4,
            "obs":2,"act":1, "hash":"frozen", "manifest":{}, "deploy":{}}, policy, tmp_path / "outputs")


@pytest.mark.asyncio
async def test_collector_never_registers_skipped_checks(monkeypatch, tmp_path):
    from eb.bundle.jobs import collect_bundle
    from eb.store.ebd import EbdClient
    monkeypatch.setattr("eb.bundle.jobs.get_settings", lambda: NS(data_root=tmp_path))
    read = AsyncMock(return_value=json.dumps({"status":"complete", "numerical_check":{"passed":False}}))
    monkeypatch.setattr(EbdClient, "read_storage_text", read)
    packed = AsyncMock()
    monkeypatch.setattr(EbdClient, "pack_bundle", packed, raising=False)
    session = NS(add=lambda _: pytest.fail("registered invalid bundle"))
    with pytest.raises(ValueError):
        await collect_bundle(session, NS(spec={"params":{"bundle_plan":{"hash":"frozen"}}}, embodiment_env_id="env"), NS(id="run_bundle"))
    packed.assert_not_awaited()


@pytest.mark.asyncio
async def test_bundle_freezes_registered_bytes_and_rejects_g4(tmp_path, monkeypatch):
    from eb.bundle.jobs import freeze_bundle, validate_bundle_plan
    from eb.domain.contract import load_env
    from eb.domain.job import JobSpec
    from eb.store.models import Artifact, ModelVersion
    from eb.store.ebd import EbdClient
    env = load_env(Path(__file__).resolve().parents[3] / "contracts/ee_bhl_biped_12dof_v1.yaml")
    artifact = Artifact(id="art_policy", kind="onnx", storage=str(tmp_path / "policy.onnx"), sha256="a"*64, size_bytes=4)
    model = ModelVersion(id="mv_policy", embodiment_env_id=env.id, artifact_id=artifact.id, card={})
    session = NS(get=AsyncMock(side_effect=lambda cls,_: model if cls is ModelVersion else artifact))
    monkeypatch.setattr("eb.bundle.jobs.storage_root", lambda _: {"namespace":"state"})
    monkeypatch.setattr(EbdClient, "hash_storage_file", AsyncMock(return_value={"sha256":"a"*64,"bytes":4}))
    monkeypatch.setattr(EbdClient, "image_info", AsyncMock(return_value={"ok":True,"digest":"sha256:"+"b"*64}))
    spec = JobSpec(kind="bundle.build", embodiment_env_id=env.id, inputs={"base_model":model.id})
    frozen = await freeze_bundle(session, spec, env)
    assert frozen.params["bundle_plan"]["manifest"]["safety"] == {
        "action_clip":env.action.scale * 1.5,"max_cmd_vel":[1.0,0.5,0.8],"watchdog_ms":100,"fallback":"freeze_or_damp"}
    assert frozen.params["bundle_plan"]["manifest"]["g3"]["passed"] is True
    assert frozen.runtime == {"network":"none"} and frozen.params["bundle_plan"]["onnx_sha256"] == artifact.sha256
    await validate_bundle_plan(frozen)
    bad = JobSpec(kind="bundle.build", embodiment_env_id=env.id, inputs={"base_model":model.id}, params={"deploy":{"num_actions":1}})
    with pytest.raises(ValueError, match="G4"):
        await freeze_bundle(session, bad, env)


@pytest.mark.asyncio
async def test_checked_bundle_registration_uses_rust_publication_and_stays_unsigned(tmp_path, monkeypatch):
    import hashlib
    from eb.bundle.jobs import collect_bundle, fingerprint
    from eb.bundle.worker import stage_bundle
    from eb.store.ebd import EbdClient
    from eb.store.models import Bundle
    policy = tmp_path / "policy.onnx"; policy.write_bytes(b"fake")
    monkeypatch.setattr("eb.bundle.worker.read_signature", lambda _: {"obs":2,"act":1})
    monkeypatch.setattr("eb.bundle.worker.numerical_check", lambda *_: {"passed":True,"provider":"CPUExecutionProvider"})
    plan = {"model_version_id":"mv_policy","onnx_sha256":hashlib.sha256(b"fake").hexdigest(),"onnx_bytes":4,
            "obs":2,"act":1,"manifest":{"embodiment_env_id":"env","gates_passed":["G3","G4","numerical_check"]}, "deploy":{}}
    plan["hash"] = fingerprint(plan)
    outputs = tmp_path / "outputs"; outputs.mkdir()
    stage_bundle(plan, policy, outputs)
    async def text(root, path, **_):
        return (outputs / ("bundle-staging" if root["path"].endswith("bundle-staging") else "") / path).read_text()
    monkeypatch.setattr(EbdClient, "read_storage_text", AsyncMock(side_effect=text))
    publication = {"sha256":"c"*64,"size_bytes":100,"storage":"/cas/cc/"+"c"*64}
    packed = AsyncMock(return_value=publication); monkeypatch.setattr(EbdClient, "pack_bundle", packed)
    rows = []; session = NS(scalar=AsyncMock(return_value=None), add=rows.append)
    run = NS(id="run_bundle")
    await collect_bundle(session, NS(spec={"params":{"bundle_plan":plan}},embodiment_env_id="env"), run)
    bundle = next(row for row in rows if isinstance(row, Bundle))
    assert bundle.signed is False and bundle.manifest["can_sign"] is False
    assert run.metrics_summary["bundle_id"] == bundle.id and packed.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["root", "numerical", "members"])
async def test_malformed_bundle_receipt_is_a_content_failure(monkeypatch, malformed):
    from eb.bundle.jobs import collect_bundle, fingerprint
    from eb.store.ebd import EbdClient
    plan={"obs":2,"act":1,"manifest":{"gates_passed":["G3","G4","numerical_check"]}}
    plan["hash"]=fingerprint(plan)
    receipt={"schema_version":"eb/bundle-checks/v1","status":"complete","plan_hash":plan["hash"],
             "numerical_check":{"passed":True},"io_signature":{"obs":2,"act":1},
             "manifest":{**plan["manifest"],"numerical_check":{"passed":True}},"members":[1]*5}
    if malformed=="root": receipt=[]
    elif malformed=="numerical": receipt["numerical_check"]=None
    monkeypatch.setattr(EbdClient, "read_storage_text", AsyncMock(return_value=json.dumps(receipt)))
    with pytest.raises(ValueError):
        await collect_bundle(NS(),NS(spec={"params":{"bundle_plan":plan}}),NS(id="run_test"))
