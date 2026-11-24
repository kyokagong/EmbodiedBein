"""Run in eb/motiondecode-preview with G1 assets mounted read-only at /robot."""

import hashlib
import json
import tempfile
import time
from pathlib import Path

from eb.motiondecode import HEADER
from eb.motiondecode_preview import render

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "motion.csv"
    row = ",".join(map(str, [0, 0, 0.8, 1, 0, 0, 0] + [0] * 29)) + "\n"
    source.write_text(",".join(HEADER) + "\n" + row * 64)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    started = time.monotonic()
    receipt = render(
        source, digest, Path("/robot"), 0, root / "outputs", source_path="fixture.csv", asset_commit="fixture"
    )
    assert receipt["video_frames"] == 16 and receipt["frames"] == 64
    assert receipt["kinematic_only"] is True and receipt["dynamics_validated"] is False
    table = json.loads((root / "outputs/table/000000.json").read_text())
    assert table["rows"][0][:7] == [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    assert len(table["columns"]) == 36
    print("G1_POSE_TABLE_AND_DECODED_MP4_OK", {"video_frames": 16, "elapsed_s": time.monotonic() - started})
