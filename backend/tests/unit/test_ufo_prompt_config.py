"""The published task image must receive native RewardInference's agent config."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eb.ufo_native import plan_hash
from eb.ufo_training_prompt import INFERENCE_FILES


def frozen_prompt_source(tmp_path, monkeypatch):
    from eb import ufo_training_prompt as service

    native = tmp_path / "native"
    for relative in (*INFERENCE_FILES, "checkpoint/config.json", "checkpoint-manifest.json"):
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": {"name": "FBcprAuxModel"}}))
    xml, robot, motion, license_file = [tmp_path / name for name in ("rpo.xml", "robot.yaml", "train.pkl", "LICENSE")]
    xml.write_text('<mujoco><compiler meshdir="meshes"/></mujoco>')
    robot.write_text("xml_path: rpo.xml\n")
    motion.write_bytes(b"frozen native training data")
    license_file.write_text("UFO license")
    (native / "config.json").write_text(json.dumps({"env": {
        "mjcf_path": str(xml), "robot_config_path": str(robot)}}))
    digest = "sha256:" + "a" * 64
    refs = {name: {"path": f"/sources/{name}", "manifest_path": f"/sources/{name}-manifest.json",
                   "snapshot_hash": digest} for name in ("ufo", "rlinf")}
    source_ref = {"kind": "content_snapshot", "path": str(tmp_path), "commit": digest}
    checkpoint = {"native_path": str(native), "run_id": "run_training", "image_digest": digest,
                  "source_refs": refs, "compatibility_fingerprint": digest, "replay_id": "snapshot-1",
                  "replay": [{"snapshot": {"id": "snapshot-1", "transition_format": 2}}],
                  "files": [{"path": relative, "bytes": (native / relative).stat().st_size,
                             "sha256": hashlib.sha256((native / relative).read_bytes()).hexdigest()}
                            for relative in (*INFERENCE_FILES, "checkpoint/config.json")]}
    model = NS(id="mv_trained", name="Native trained", card={"codec": {}, "model_sha256": "b" * 64})
    job = NS(spec={"params": {"ufo_source_ref": source_ref}, "code_refs": {"stack": source_ref},
                   "inputs": {"datasets": [{"path": str(tmp_path)}]}})
    monkeypatch.setattr(service, "trained_source", AsyncMock(return_value=(
        model, NS(sha256=digest), NS(sha256=digest), job, checkpoint)))
    monkeypatch.setattr(service, "source_path", lambda value, *_: Path(value))

    async def scanned_file(path):
        return {"source": str(path), "path": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    monkeypatch.setattr("eb.api.routes_ufo_native.scanned_file", scanned_file)
    client = NS(image_info=AsyncMock(return_value={"ok": True, "digest": digest}))
    monkeypatch.setattr("eb.store.ebd.EbdClient", lambda: client)
    return native, checkpoint


@pytest.mark.asyncio
async def test_prompt_plan_stages_native_agent_config_without_changing_frozen_runner_files(tmp_path, monkeypatch):
    from eb.api.routes_ufo_native import spec_for
    from eb.ufo_training_prompt import prompt_plan, render_prompt

    native, checkpoint = frozen_prompt_source(tmp_path, monkeypatch)
    plan = await prompt_plan(NS(), "mv_trained")
    from eb.store.ebd import EbdClient
    EbdClient().image_info.assert_awaited_once_with(checkpoint["image_digest"])
    extra = next(item for item in plan["extras"] if item["path"] == "checkpoint/config.json")
    original = next(item for item in checkpoint["files"] if item["path"] == extra["path"])
    assert extra == {**original, "source": str(native / "checkpoint/config.json")}
    assert {item["path"] for item in plan["files"]} == set(INFERENCE_FILES)
    assert plan["hash"] == plan_hash(plan)
    assert not any("optimizers" in item["path"] or "buffers/" in item["path"]
                   for item in plan["files"] + plan["extras"])
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path / "state"))
    rendered = render_prompt(spec_for(plan), "run_preview")
    index = plan["extras"].index(extra)
    assert (extra["source"], f"/source-extra/{index}", "ro") in rendered.mounts


@pytest.mark.asyncio
async def test_prompt_agent_config_rejects_symlink_source(tmp_path, monkeypatch):
    from eb.ufo_training_prompt import prompt_plan

    native, _ = frozen_prompt_source(tmp_path, monkeypatch)
    config = native / "checkpoint/config.json"
    config.unlink()
    config.symlink_to(native / "checkpoint/model/config.json")
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        await prompt_plan(NS(), "mv_trained")
