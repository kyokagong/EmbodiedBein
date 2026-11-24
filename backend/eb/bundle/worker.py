"""CPU model checks and explicit bundle members, executed only as a tool Job."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from eb.onnx_io import read_signature, sha256_file

MEMBERS = {"manifest.yaml", "policy.onnx", "deploy.yaml", "policy_meta.yaml", "gate_report.json"}


def numerical_check(path: Path, obs_dim: int) -> dict:
    try:
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: np.zeros((1, obs_dim), dtype=np.float32)})
        if not outputs or any(not np.isfinite(value).all() for value in outputs):
            raise ValueError("non-finite model output")
        return {"passed": True, "provider": "CPUExecutionProvider"}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def stage_bundle(plan: dict, policy: Path, outputs: Path) -> dict:
    staging = outputs / "bundle-staging"
    staging.mkdir(parents=True, exist_ok=False)
    copied = staging / "policy.onnx"
    with policy.open("rb") as source, copied.open("xb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if copied.stat().st_size != plan["onnx_bytes"] or sha256_file(copied) != plan["onnx_sha256"]:
        raise ValueError("bundle frozen policy identity differs")
    signature = read_signature(copied)
    if signature.get("obs") != plan["obs"] or signature.get("act") != plan["act"]:
        raise ValueError("bundle G3 model dimensions differ")
    numerical = numerical_check(copied, plan["obs"])
    if numerical.get("passed") is not True:
        raise ValueError("bundle numerical check failed: " + str(numerical.get("error")))
    manifest = {**plan["manifest"], "numerical_check": numerical, "gates_passed": ["G3", "G4", "numerical_check"]}
    texts = {"manifest.yaml": yaml.safe_dump(manifest), "deploy.yaml": yaml.safe_dump(plan["deploy"]),
             "policy_meta.yaml": yaml.safe_dump(plan["deploy"]),
             "gate_report.json": yaml.safe_dump({"manifest": manifest})}
    for name, text in texts.items():
        (staging / name).write_text(text)
    members = [{"path": name, "bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name)} for name in sorted(MEMBERS)]
    receipt = {"schema_version":"eb/bundle-checks/v1", "status":"complete", "plan_hash":plan["hash"],
               "numerical_check":numerical, "io_signature":{"obs":signature["obs"],"act":signature["act"]},
               "manifest":manifest,"members":members}
    (outputs / "bundle-checks.json").write_text(json.dumps(receipt, sort_keys=True))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    stage_bundle(json.loads(args.plan), Path(args.policy), Path(args.outputs))


if __name__ == "__main__":
    main()
