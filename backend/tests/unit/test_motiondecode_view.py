import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.motiondecode import HEADER
from eb.motiondecode_view import EXCLUDED_PATH, EXCLUDED_SHA256, derive_view


def test_derived_view_reuses_verified_evidence_and_keeps_raw_bytes(tmp_path):
    root = tmp_path / "raw"
    valid = root / "samples/walk.csv"
    valid.parent.mkdir(parents=True)
    valid.write_text(",".join(HEADER) + "\n" + ",".join(map(str, [0, 0, 1, 1, 0, 0, 0] + [0] * 29)) + "\n")
    # The exclusion is pinned to the exact real upstream bytes, not any empty CSV.
    empty = root / EXCLUDED_PATH
    empty.parent.mkdir(parents=True)
    empty.write_bytes((",".join(HEADER) + "\r\n").encode())
    assert hashlib.sha256(empty.read_bytes()).hexdigest() == EXCLUDED_SHA256
    before = empty.read_bytes()
    files, receipts = [], []
    for path in [valid, empty]:
        data = path.read_bytes()
        entry = {
            "path": path.relative_to(root).as_posix(),
            "size": len(data),
            "blob_id": "unused",
            "lfs_sha256": hashlib.sha256(data).hexdigest(),
        }
        files.append(entry)
        receipts.append(
            {
                **entry,
                "sha256": entry["lfs_sha256"],
                **({"error": "ValueError: empty motion"} if path == empty else {"length": 1, "columns": 36}),
            }
        )
    index = tmp_path / "file-index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    report = derive_view(root, {"files": files}, index, tmp_path / "outputs")
    assert report["validation"]["status"] == "validated"
    assert report["validation"]["preview_ready"] is True
    assert report["validation"]["training_ready"] is False
    assert report["episodes_count"] == 1
    assert report["exclusions"][0]["sha256"] == EXCLUDED_SHA256
    assert empty.read_bytes() == before
    valid.write_bytes(valid.read_bytes().replace(b"0,0,1", b"0,0,2"))
    with pytest.raises(ValueError, match="hash"):
        derive_view(root, {"files": files}, index, tmp_path / "bad")


def test_truncated_source_evidence_cannot_publish_a_view(tmp_path):
    index = tmp_path / "index.jsonl"
    index.write_text("")
    with pytest.raises(ValueError, match="evidence"):
        derive_view(tmp_path, {"files": [{"path": "samples/a.csv", "size": 1}]}, index, tmp_path / "out")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,has_run", [("timeout", True), ("timeout", False), ("cancelled", False), ("blocked", False)]
)
async def test_derived_view_terminal_jobs_remain_retryable(status, has_run):
    from eb.api.routes_data import reconcile_dataset
    from eb.store.models import Dataset

    row = Dataset(
        id="ds_view",
        source_kind="motiondecode_view",
        source_uri="CMRobot/MotionDecode",
        status="queued",
        meta={"validation_job_id": "job_view"},
    )
    job = SimpleNamespace(id="job_view", status=status, error="stopped before publication")
    run = SimpleNamespace(id="run_view") if has_run else None
    session = SimpleNamespace(
        get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=run), commit=AsyncMock()
    )
    await reconcile_dataset(row, session)
    assert row.status == "failed"
    assert row.error_message == "stopped before publication"
