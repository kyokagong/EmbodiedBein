from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _last_dim(vi) -> int | None:
    dims = vi.type.tensor_type.shape.dim
    if not dims:
        return None
    d = dims[-1]
    if d.dim_value:
        return int(d.dim_value)
    return None


def read_signature(path: Path) -> dict[str, Any]:
    """Measure ONNX graph I/O last-dim. Filename is never used as a pass criterion."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    if not model.graph.input or not model.graph.output:
        raise ValueError(f"onnx has no graph I/O: {path}")
    obs = _last_dim(model.graph.input[0])
    act = _last_dim(model.graph.output[0])
    return {
        "obs": obs,
        "act": act,
        "n_inputs": len(model.graph.input),
        "n_outputs": len(model.graph.output),
        "path": str(path),
    }
