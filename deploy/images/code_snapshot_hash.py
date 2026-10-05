#!/usr/bin/env python3
"""Print the same code-tree hash as ebd's snapshot service."""
import hashlib
import json
import sys
from pathlib import Path

EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            ".venv", ".claude", ".agents", "CLAUDE.md", "node_modules", ".cache"}
ROOT_OUTPUTS = {"target", "logs", "outputs", "checkpoints", "datasets", "venv"}
WEIGHT_EXTENSIONS = {"safetensors", "pt", "pth", "ckpt", "onnx", "gguf", "msgpack", "h5"}
WEIGHT_BIN_PREFIXES = ("pytorch_model", "model", "adapter_model")


def code_snapshot_hash(root: Path) -> str:
    root = root.resolve(strict=True)
    entries = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda value: value.name):
            if path.name in EXCLUDED or (directory == root and path.name in ROOT_OUTPUTS):
                continue
            if path.is_symlink():
                raise ValueError(f"snapshot symlink forbidden: {path.relative_to(root)}")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                entries.append({"path": relative, "sha256": None, "executable": False})
                visit(path)
                continue
            suffix = path.suffix.removeprefix(".").lower()
            if suffix in WEIGHT_EXTENSIONS or (suffix == "bin" and path.name.lower().startswith(WEIGHT_BIN_PREFIXES)):
                continue
            if not path.is_file():
                raise ValueError(f"special file forbidden: {relative}")
            entries.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "executable": bool(path.stat().st_mode & 0o111)})

    visit(root)
    manifest = {"version": 2, "entries": entries, "exclusions": {
        "directories": [".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                        ".venv", ".claude", ".agents", "CLAUDE.md", "node_modules", ".cache"],
        "root_outputs": ["target", "logs", "outputs", "checkpoints", "datasets", "venv"],
        "weight_extensions": ["safetensors", "pt", "pth", "ckpt", "onnx", "gguf", "msgpack", "h5"],
        "weight_bin_prefixes": ["pytorch_model", "model", "adapter_model"],
        "weight_extension_case": "ascii_case_insensitive",
    }}
    payload = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False,
                         sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    print(code_snapshot_hash(Path(sys.argv[1])))
