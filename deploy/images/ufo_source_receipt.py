"""Verify pinned Git blobs before promoting agent text into a managed snapshot."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from code_snapshot_hash import code_snapshot_hash

REVISION = "0b89378de02f77712072d983bc43ec2a46455dc0"
TREE = "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9"


def git_hash(kind, payload):
    return hashlib.sha1(f"{kind} {len(payload)}\0".encode() + payload).hexdigest()


def verify_snapshot(root, tree):
    entries = tree["tree"]
    if tree.get("sha") not in {TREE, REVISION} or tree.get("truncated"):
        raise ValueError("Pinned UFO tree missing or truncated")
    by_directory = {}
    for entry in entries:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe source tree path")
        by_directory.setdefault(path.parent.as_posix(), []).append(entry)
    for directory, children in by_directory.items():
        payload = b""
        for entry in sorted(children, key=lambda e: Path(e["path"]).name + ("/" if e["type"] == "tree" else "")):
            payload += (entry["mode"].lstrip("0") + " " + Path(entry["path"]).name).encode() + b"\0" + bytes.fromhex(entry["sha"])
        expected = TREE if directory == "." else next(e["sha"] for e in entries if e["path"] == directory)
        if git_hash("tree", payload) != expected:
            raise ValueError(f"Git tree hash differs: {directory}")
    selected = [e for e in entries if e["type"] == "blob" and
        (e["path"].startswith("humanoidverse/") and e["path"].endswith(".py")
         or e["path"] in {"LICENSE", "pyproject.toml"})]
    for entry in selected:
        path = root / entry["path"]
        if path.is_symlink() or not path.is_file() or git_hash("blob", path.read_bytes()) != entry["sha"]:
            raise ValueError(f"Git blob differs from pinned source: {entry['path']}")
    return selected


def prepare(source, tree_path, destination):
    tree = json.loads(tree_path.read_text())
    selected = verify_snapshot(source, tree)
    if destination.exists():
        verify_snapshot(destination, tree)
        return
    staging = destination.with_name(destination.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    for entry in selected:
        target = staging / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / entry["path"], target)
    (staging / ".eb-upstream-tree.json").write_text(json.dumps(tree, sort_keys=True))
    staging.rename(destination)


def image_receipt(revision, content_hash):
    destination = Path("/opt/ufo")
    verify_snapshot(destination, json.loads((destination / ".eb-upstream-tree.json").read_text()))
    base = json.loads(Path("/workspace/rlinf/.eb-source-receipt.json").read_text())
    if base["rlinf_content_hash"] != content_hash:
        raise ValueError("RLinf base snapshot differs from current read-only source")
    receipt = {"schema_version": "eb/ufo-image-source/v1",
        "ufo": {"commit": REVISION, "tree": TREE, "content_hash": code_snapshot_hash(destination),
            "license": "CC-BY-NC-4.0", "scope": "agent Python source; binary assets excluded"},
        "rlinf": {"commit": revision, "content_hash": content_hash, "license": "Apache-2.0",
            "base_patch_hash": base["patch_sha256"]},
        "patch_hash": "sha256:" + hashlib.sha256(Path("/tmp/patch_rpo_ufo.py").read_bytes()).hexdigest(),
        "packages": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("torch", "gymnasium", "pydantic", "safetensors", "tensordict")}},
        "scope": "agent-only; Python3.10 package metadata bypassed by tested source import"}
    Path("/opt/ufo-source-receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "--prepare":
        prepare(*(Path(arg) for arg in sys.argv[2:]))
    else:
        image_receipt(*sys.argv[1:])
