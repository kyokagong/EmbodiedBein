"""Fixed, bounded G1 source acquisition; deserialize only in a later offline Job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

HF_REVISION = "40faaedd0d75a024685cfc48bd8eaff2f6772727"
UFO_REVISION = "0b89378de02f77712072d983bc43ec2a46455dc0"
SOURCES = {
    "lafan_29dof_10s-clipped.pkl": {
        "url":f"https://huggingface.co/datasets/xuewang/UFO-MotionData/resolve/{HF_REVISION}/g1/lafan_29dof_10s-clipped.pkl",
        "bytes":205117479, "sha256":"7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
        "revision":HF_REVISION, "license":"CC-BY-NC-ND-4.0 (LAFAN1 upstream)"},
    "g1_29dof.xml": {
        "url":f"https://raw.githubusercontent.com/Roboparty/UFO/{UFO_REVISION}/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml",
        "bytes":25377, "sha256":"579c8d3d98afa207cc816f1f6bdbe2594c6153cd00d6cdf644a102a4794b5920",
        "revision":UFO_REVISION, "license":"UFO CC-BY-NC-4.0; Unitree source assets retain upstream terms"},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for name, item in SOURCES.items():
        url = item["url"]
        if url.startswith("https://huggingface.co/"):
            url = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + url[len("https://huggingface.co"):]
        path = output / name
        received = 0
        with urllib.request.urlopen(url, timeout=30) as response, path.with_suffix(path.suffix+".part").open("xb") as stream:
            while chunk := response.read(1024*1024):
                received += len(chunk)
                if received > item["bytes"] or time.monotonic()-started > 600:
                    raise ValueError("G1 source exceeds fixed byte/time budget")
                stream.write(chunk)
        partial = path.with_suffix(path.suffix+".part")
        if received != item["bytes"] or sha(partial) != item["sha256"]:
            raise ValueError(f"G1 pinned source size/hash differs: {name}")
        partial.rename(path)
        print(json.dumps({"file":name, "verified_bytes":received}), flush=True)
    receipt = {"schema_version":"eb/g1-lafan-source/v1", "status":"verified",
        "source_use":"local_noncommercial_no_redistribution", "files":SOURCES,
        "motion_deserialized":False}
    (output / "source-manifest.json").write_text(json.dumps(receipt, indent=2))


def load_verified(source):
    receipt = source / "source-manifest.json"
    if receipt.is_symlink() or json.loads(receipt.read_text()).get("files") != SOURCES:
        raise ValueError("G1 source receipt differs from the fixed source profile")
    for name, item in SOURCES.items():
        path = source / name
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("G1 source bytes changed after acquisition")
    import pickle
    with (source / "lafan_29dof_10s-clipped.pkl").open("rb") as stream:
        try:
            data = pickle.load(stream)
        except pickle.UnpicklingError:
            import joblib
            data = joblib.load(source / "lafan_29dof_10s-clipped.pkl")
    if not isinstance(data, dict) or not 1 <= len(data) <= 10000:
        raise ValueError("G1 source must be a bounded multi-motion dictionary")
    return data


def inspect(source, output):
    data = load_verified(source)
    import numpy as np
    rows = []
    for name, clip in data.items():
        if not isinstance(name, str) or len(name) > 500 or not isinstance(clip, dict):
            raise ValueError("G1 source has invalid motion IDs/entries")
        rows.append({"id":name, "fields":{
            key:{"shape":list(np.shape(value)), "type":type(value).__name__}
            for key, value in clip.items()}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version":"eb/g1-lafan-inspection/v1", "clips":len(rows),
        "source_manifest_sha256":sha(source / "source-manifest.json"), "rows":rows}, indent=2))
    print(json.dumps({"clips":len(rows), "first_clip":rows[0]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("acquire", "inspect"), required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.phase == "acquire":
        acquire(args.output)
    else:
        inspect(args.source, args.output)
