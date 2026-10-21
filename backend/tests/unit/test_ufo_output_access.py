import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eb.ufo_native import plan_hash
from eb.ufo_output_access import completed_output_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def member(root, name, data=b"indexed evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def package_output(tmp_path, operation="ufo_native_prompt"):
    root = tmp_path / "runs/run_output/outputs"
    package = "payload" if operation == "ufo_native_export" else "model"
    names = ("policy.onnx", "runtime.py") if package == "payload" else (
        "prompts.json", "checkpoint/model/model.safetensors")
    files = [member(root / package, name) for name in names]
    manifest = {"files": files}
    write_json(root / package / "manifest.json", manifest)
    receipt = {"schema_version": "eb/ufo-native-receipt/v1", "run_id": "run_output",
               "operation": operation, "plan_hash": "sha256:" + "a" * 64,
               "package": package, "manifest": manifest}
    if operation == "ufo_native_prompt":
        prompts = {"files": [member(root / "prompts", "prompts.json"),
                             member(root / "prompts", "derivation.json")]}
        write_json(root / "prompts/manifest.json", prompts)
        receipt["prompt_manifest"] = prompts
    write_json(root / "receipt.json", receipt)
    return root, receipt


def resolve(root, receipt, relative):
    return completed_output_file(root, relative, run_id="run_output", operation=receipt["operation"],
                                 expected_plan_hash=receipt["plan_hash"])


@pytest.mark.parametrize("relative", ["model/prompts.json", "model/checkpoint/model/model.safetensors",
    "model/manifest.json", "prompts/prompts.json", "prompts/derivation.json", "prompts/manifest.json",
    "receipt.json"])
def test_prompt_receipt_maps_explicit_nested_package_members(tmp_path, relative):
    root, receipt = package_output(tmp_path)
    assert resolve(root, receipt, relative) == root / relative


@pytest.mark.parametrize("relative", ["payload/policy.onnx", "payload/runtime.py", "payload/manifest.json"])
def test_export_receipt_maps_the_frozen_payload_index(tmp_path, relative):
    root, receipt = package_output(tmp_path, "ufo_native_export")
    assert resolve(root, receipt, relative) == root / relative


@pytest.mark.parametrize("relative", ["model/unindexed.txt", "model/../receipt.json", "../outside",
    "/etc/passwd", "prompts/./prompts.json", "cache/weights.pt", "native/checkpoint/optimizers.pth"])
def test_unindexed_or_escaping_package_members_are_private(tmp_path, relative):
    root, receipt = package_output(tmp_path)
    member(root, "model/unindexed.txt")
    with pytest.raises(ValueError):
        resolve(root, receipt, relative)


@pytest.mark.parametrize("change", ["digest", "size", "duplicate", "path", "manifest", "symlink"])
def test_package_access_rejects_invalid_hash_index_changed_size_and_symlinks(tmp_path, change):
    root, receipt = package_output(tmp_path)
    if change == "digest":
        receipt["manifest"]["files"][0].pop("sha256")
    elif change == "size":
        (root / "model/prompts.json").write_bytes(b"different file size")
    elif change == "duplicate":
        receipt["manifest"]["files"].append(deepcopy(receipt["manifest"]["files"][0]))
    elif change == "path":
        receipt["manifest"]["files"][0]["path"] = "../receipt.json"
    elif change == "manifest":
        write_json(root / "model/manifest.json", {"files": []})
    else:
        (root / "model/prompts.json").unlink()
        (root / "model/prompts.json").symlink_to(root / "receipt.json")
    if change != "manifest":
        write_json(root / "model/manifest.json", receipt["manifest"])
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError):
        resolve(root, receipt, "model/prompts.json")


def training_output(tmp_path):
    root = tmp_path / "runs/run_output/outputs"
    names = ["analysis/training-1.png", "analysis/resources.png", "analysis/summary.json",
             "analysis/train-log-snapshot.csv", "analysis/resource-snapshot.csv",
             "native/train_log.txt", "native/ufo_tracking_eval.csv",
             "native/checkpoint/model/model.safetensors",
             "native/checkpoint/optimizers.pth", "native/checkpoint/buffers/train_rank_0/buffer.hdf5",
             "cache/model.safetensors", "analysis/model.safetensors"]
    files = [member(root, name) for name in names]
    index = {"schema_version": "eb/ufo-native-metrics-index/v1", "run_id": "run_output", "files": files}
    write_json(root / "metrics-index.json", index)
    receipt = {"schema_version": "eb/ufo-native-train-receipt/v1", "run_id": "run_output",
               "operation": "ufo_native_train", "status": "complete", "plan_hash": "sha256:" + "a" * 64}
    write_json(root / "receipt.json", receipt)
    return root, receipt


@pytest.mark.parametrize("relative", ["analysis/training-1.png", "analysis/resources.png",
    "analysis/summary.json", "analysis/train-log-snapshot.csv", "analysis/resource-snapshot.csv",
    "native/train_log.txt", "native/ufo_tracking_eval.csv", "metrics-index.json", "receipt.json"])
def test_native_train_serves_only_bounded_analysis_evidence(tmp_path, relative):
    root, receipt = training_output(tmp_path)
    assert resolve(root, receipt, relative) == root / relative


@pytest.mark.parametrize("relative", ["native/checkpoint/model/model.safetensors",
    "native/checkpoint/optimizers.pth", "native/checkpoint/buffers/train_rank_0/buffer.hdf5",
    "cache/model.safetensors", "analysis/model.safetensors", "analysis/training-2.png"])
def test_native_train_refuses_private_or_unindexed_state_even_if_other_state_is_indexed(tmp_path, relative):
    root, receipt = training_output(tmp_path)
    with pytest.raises(ValueError):
        resolve(root, receipt, relative)


@pytest.mark.parametrize("relative", ["native/train_log.txt", "native/ufo_tracking_eval.csv"])
def test_native_original_metrics_require_a_hash_index_entry(tmp_path, relative):
    root, receipt = training_output(tmp_path)
    index = json.loads((root / "metrics-index.json").read_text())
    index["files"] = [item for item in index["files"] if item["path"] != relative]
    write_json(root / "metrics-index.json", index)
    with pytest.raises(ValueError, match="hash index"):
        resolve(root, receipt, relative)


def test_native_train_checks_index_identity_and_completion_before_reading_evidence(tmp_path):
    root, receipt = training_output(tmp_path)
    index = json.loads((root / "metrics-index.json").read_text())
    index["run_id"] = "run_other"
    write_json(root / "metrics-index.json", index)
    with pytest.raises(ValueError, match="identity"):
        resolve(root, receipt, "analysis/summary.json")
    receipt["status"] = "partial"
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError, match="identity"):
        resolve(root, receipt, "receipt.json")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,relative", [
    ("ufo_native_prompt", "prompts/prompts.json"), ("ufo_native_export", "payload/policy.onnx"),
    ("ufo_native_train", "analysis/training-1.png"),
])
async def test_file_route_streams_indexed_nested_and_native_train_outputs_without_new_jobs(
    tmp_path, monkeypatch, operation, relative,
):
    from eb.api.routes_ufo_native import output_file

    root, receipt = (training_output(tmp_path) if operation == "ufo_native_train"
                     else package_output(tmp_path, operation))
    params = {"operation": operation, "native_plan": {"hash": receipt["plan_hash"]}}
    if operation == "ufo_native_train":
        params = {"template_id": "rpo_ufo_fb_native", "native_train_plan": {"fingerprint": "frozen"}}
        receipt["plan_hash"] = plan_hash({**params["native_train_plan"], "run_id": "run_output"})
        write_json(root / "receipt.json", receipt)
    job = NS(kind="train.rl" if operation == "ufo_native_train" else "data.convert", spec={"params": params})
    run = NS(id="run_output", job_id="job_output", status="finished")
    session = NS(get=AsyncMock(side_effect=[run, job]))
    stream = AsyncMock(return_value="streamed")
    monkeypatch.setattr("eb.api.routes_ufo_native.get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr("eb.api.routes_ufo_native.stream_output", stream)
    request = NS(method="HEAD", headers={"range": "bytes=0-15"})
    assert await output_file("run_output", relative, request, session, None) == "streamed"
    stream.assert_awaited_once_with(request, "run_output", relative)


@pytest.mark.asyncio
async def test_running_native_train_cannot_stream_partial_analysis(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from eb.api.routes_ufo_native import output_file

    training_output(tmp_path)
    job = NS(kind="train.rl", spec={"params": {"template_id": "rpo_ufo_fb_native"}})
    session = NS(get=AsyncMock(side_effect=[NS(job_id="job_output", status="running"), job]))
    stream = AsyncMock()
    monkeypatch.setattr("eb.api.routes_ufo_native.get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr("eb.api.routes_ufo_native.stream_output", stream)
    with pytest.raises(HTTPException) as denied:
        await output_file("run_output", "analysis/training-1.png", NS(), session, None)
    assert denied.value.status_code == 404
    stream.assert_not_awaited()
