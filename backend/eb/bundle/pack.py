from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import yaml

from eb.gates.engine import g3_model, g4_export
from eb.onnx_io import read_signature
from eb.settings import get_settings


def build_ebp(
    *,
    contract_id: str,
    onnx_path: Path,
    deploy_yaml: dict[str, Any],
    gates_passed: list[str],
    provenance: dict[str, Any],
    ctr,
) -> tuple[Path, dict[str, Any], bool]:
    """Pack tar.zst .ebp into CAS. signed=False unless G3+G4 pass (caller adds sim2sim)."""
    sig = read_signature(onnx_path)
    g3 = g3_model(sig.get("obs"), sig.get("act"), ctr)
    g4 = g4_export(deploy_yaml, ctr)
    passed = list(gates_passed)
    if g3.passed and "G3" not in passed:
        passed.append("G3")
    if g4.passed and "G4" not in passed:
        passed.append("G4")
    can_sign = g3.passed and g4.passed and "sim2sim" in {x.lower() for x in passed} or (
        g3.passed and g4.passed and "G-sim2sim" in passed
    )
    # require explicit sim2sim in gates_passed
    can_sign = g3.passed and g4.passed and any("sim2sim" in x.lower() for x in passed)
    manifest = {
        "contract_id": contract_id,
        "io_signature": {"obs": sig.get("obs"), "act": sig.get("act")},
        "gates_passed": passed,
        "provenance": provenance,
        "g3": g3.model_dump(),
        "g4": g4.model_dump(),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = yaml.safe_dump(manifest).encode()
        info = tarfile.TarInfo("manifest.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        onnx_bytes = onnx_path.read_bytes()
        info = tarfile.TarInfo("policy.onnx")
        info.size = len(onnx_bytes)
        tar.addfile(info, io.BytesIO(onnx_bytes))
        dep = yaml.safe_dump(deploy_yaml).encode()
        info = tarfile.TarInfo("deploy.yaml")
        info.size = len(dep)
        tar.addfile(info, io.BytesIO(dep))
    raw = buf.getvalue()
    try:
        import pyzstd

        packed = pyzstd.compress(raw)
    except Exception:
        packed = raw
    digest = hashlib.sha256(packed).hexdigest()
    dest_dir = get_settings().data_root / "cas" / digest[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest}.ebp"
    dest.write_bytes(packed)
    return dest, manifest, can_sign
