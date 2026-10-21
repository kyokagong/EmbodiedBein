"""Bounded public evidence paths; large-file digests are verified by ebd at collection."""

import re
from pathlib import Path, PurePosixPath

from eb.ufo_native import read_json, safe_file

MAX_INDEX_FILES = 2048
MAX_PACKAGE_BYTES = 2 * 1024**3
MAX_EVIDENCE_BYTES = 64 * 1024**2
TRAINING_EVIDENCE = re.compile(
    r"(?:analysis/(?:training-[1-9][0-9]*\.png|resources\.png|summary\.json|"
    r"train-log-snapshot\.csv|resource-snapshot\.csv)|native/(?:train_log\.txt|ufo_tracking_eval\.csv))"
)


def _relative(value):
    if (not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Unsafe native evidence path")
    return value


def _root(root):
    root = Path(root)
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("Native evidence root must not contain symlinks")
    return root


def _records(files):
    if not isinstance(files, list) or len(files) > MAX_INDEX_FILES:
        raise ValueError("Native evidence index exceeds its bounded file count")
    result = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid native evidence file record")
        relative = _relative(item.get("path"))
        if (relative in result or type(item.get("bytes")) is not int
                or not 0 <= item["bytes"] <= MAX_PACKAGE_BYTES
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Native evidence file index lacks unique size/hash metadata")
        result[relative] = item
    return result


def indexed_output_file(root, relative, files, *, max_bytes=MAX_PACKAGE_BYTES):
    root = _root(root)
    relative = _relative(relative)
    record = _records(files).get(relative)
    if record is None or record["bytes"] > max_bytes:
        raise ValueError("Native evidence file is not in its bounded hash index")
    path = safe_file(root, relative)
    if path.stat().st_size != record["bytes"]:
        raise ValueError("Native evidence file size changed after publication")
    return path


def _package_file(root, relative, package, manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Missing native output package manifest")
    package_root = _root(root / package)
    actual = read_json(safe_file(package_root, "manifest.json"), limit=4 * 1024**2)
    if actual != manifest:
        raise ValueError("Native output package differs from its completion receipt")
    member = relative.removeprefix(package + "/")
    if member == "manifest.json":
        _records(manifest.get("files"))
        return safe_file(package_root, member)
    if any(part in {"buffers", ".saving", "optimizers.pth"} for part in PurePosixPath(member).parts):
        raise ValueError("Native training state is not a public inference package member")
    return indexed_output_file(package_root, member, manifest.get("files"))


def completed_output_file(root, relative, *, run_id, operation, expected_plan_hash):
    root = _root(root)
    relative = _relative(relative)
    receipt_path = safe_file(root, "receipt.json")
    receipt = read_json(receipt_path, limit=4 * 1024**2)
    native_train = operation == "ufo_native_train"
    schema = "eb/ufo-native-train-receipt/v1" if native_train else "eb/ufo-native-receipt/v1"
    if (receipt.get("schema_version") != schema or receipt.get("run_id") != run_id
            or receipt.get("operation") != operation
            or not isinstance(expected_plan_hash, str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_plan_hash)
            or receipt.get("plan_hash") != expected_plan_hash
            or native_train and receipt.get("status") != "complete"):
        raise ValueError("Native output completion receipt identity differs")
    if relative == "receipt.json":
        return receipt_path
    if native_train:
        index_path = safe_file(root, "metrics-index.json")
        index = read_json(index_path, limit=1024**2)
        if (index.get("schema_version") != "eb/ufo-native-metrics-index/v1"
                or index.get("run_id") != run_id):
            raise ValueError("Native training evidence index identity differs")
        _records(index.get("files"))
        if relative == "metrics-index.json":
            return index_path
        if not TRAINING_EVIDENCE.fullmatch(relative):
            raise ValueError("Only indexed analysis and original metric files are public for native training")
        return indexed_output_file(root, relative, index["files"], max_bytes=MAX_EVIDENCE_BYTES)
    if operation in {"ufo_native_import", "ufo_native_prompt", "ufo_native_export"}:
        package = "payload" if operation == "ufo_native_export" else "model"
        if receipt.get("package") != package:
            raise ValueError("Native output package identity differs")
        if relative.startswith(package + "/"):
            return _package_file(root, relative, package, receipt.get("manifest"))
        if operation == "ufo_native_prompt" and relative.startswith("prompts/"):
            return _package_file(root, relative, "prompts", receipt.get("prompt_manifest"))
        raise ValueError("Native output is outside its published inference or prompt package")
    return indexed_output_file(root, relative, receipt.get("files"))
