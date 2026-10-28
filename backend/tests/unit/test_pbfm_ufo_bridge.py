from copy import deepcopy

import pytest

from eb.pbfm_ufo_bridge import grouped_source_manifest, select_grouped_records, validate_grouped_source


def inventory():
    return [
        {
            "path": "data/train.pkl",
            "sha256": "a" * 64,
            "bytes": 100,
            "parent_split": "train",
            "available_groups": ["walk2", "official_rpo_unknown_ancestry"],
        },
        {
            "path": "holdout/validation.pkl",
            "sha256": "b" * 64,
            "bytes": 100,
            "parent_split": "validation",
            "available_groups": ["walk1", "walk4"],
        },
    ]


def manifest():
    return grouped_source_manifest(
        [f"j{i}" for i in range(23)],
        inventory(),
        {"train": ["walk2"], "validation": ["walk1"], "acceptance": ["walk4"]},
        ["official_rpo_unknown_ancestry"],
    )


def test_grouped_ufo_source_keeps_original_holdout_out_of_training():
    value = manifest()
    validate_grouped_source(value)
    assert value["splits"]["validation"][0]["path"] == value["splits"]["acceptance"][0]["path"]
    assert value["splits"]["validation"][0]["ancestor_ids"] == ["walk1"]
    for groups, excluded in [
        (
            {"train": ["walk1"], "validation": ["walk2"], "acceptance": ["walk4"]},
            ["official_rpo_unknown_ancestry"],
        ),
        (
            {"train": ["walk2"], "validation": ["walk1"], "acceptance": ["walk1"]},
            ["official_rpo_unknown_ancestry", "walk4"],
        ),
        (
            {"train": ["walk2"], "validation": ["walk1"], "acceptance": ["official_rpo_unknown_ancestry"]},
            ["walk4"],
        ),
        ({"train": ["walk2"], "validation": ["walk1"], "acceptance": ["walk4"]}, []),
    ]:
        with pytest.raises(ValueError):
            grouped_source_manifest(value["joint_order"], inventory(), groups, excluded)


def test_runtime_selects_complete_groups_and_keeps_source_records_immutable():
    records = {
        group: {
            "metadata": {
                "source_group": group,
                "motion_key": group,
                "source_sha256": "c" * 64,
                "source_path": "source/" + group,
            }
        }
        for group in ("walk1", "walk4")
    }
    before = deepcopy(records)
    row = manifest()["splits"]["validation"][0]
    selected = select_grouped_records(records, row, "validation")
    assert set(selected) == {"walk1"}
    assert records == before
    assert selected["walk1"]["metadata"]["ancestor_id"] == "walk1"
    with pytest.raises(ValueError, match="inventory"):
        select_grouped_records({"walk1": records["walk1"]}, row, "validation")


@pytest.mark.asyncio
async def test_registered_orm_plan_uses_only_frozen_producer_bytes_and_rejects_changes(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from eb.api import routes_pbfm
    from eb.pbfm_data import render, spec_for, validate_data_plan
    from eb.pbfm_io import sha256, write_json
    from eb.pbfm_ufo_bridge import build_registered_ufo_plan
    from eb.store.ebd import EbdClient
    from eb.store.models import Dataset, DatasetVersion, Job, Run
    from eb.ufo_native import ENV_ID

    root = tmp_path / "runs/run_source/outputs"
    for dirname, filename, split, groups, payload in (
        ("data", "train.pkl", "train", ["walk2", "official_rpo_unknown_ancestry"], b"opaque train"),
        ("holdout", "validation.pkl", "validation", ["walk1", "walk4"], b"opaque holdout"),
    ):
        path = root / dirname
        path.mkdir(parents=True)
        (path / filename).write_bytes(payload)
        write_json(
            path / "manifest.json",
            {
                "schema_version": "eb/ufo-native-motion/v1",
                "dataset_id": "ds_source",
                "split": split,
                "train_file" if split == "train" else "validation_file": filename,
                "file_sha256": sha256(path / filename),
                "joint_order": [f"j{i}" for i in range(23)],
                "groups": groups,
                "episodes": [{"source_group": g} for g in groups],
            },
        )
    validation = {
        "scope": "ufo_native_motion_v1",
        "training_ready": True,
        "storage_ref": str(root / "data"),
        "manifest_sha256": sha256(root / "data/manifest.json"),
    }
    parent = Dataset(
        id="ds_source",
        source_kind="ufo_native_motion",
        status="validated",
        embodiment_env_id=ENV_ID,
        dest_path=str(root / "data"),
        summary_stats={"validation": validation},
        meta={"validation_run_id": "run_source", "validation_job_id": "job_source"},
    )
    version = DatasetVersion(
        id="dsv_source",
        dataset_id=parent.id,
        version="native-v1",
        content_hash="sha256:" + validation["manifest_sha256"],
        stats={
            "producer_run": "run_source",
            "holdout": {"manifest_sha256": sha256(root / "holdout/manifest.json")},
        },
    )
    run = Run(id="run_source", job_id="job_source", status="finished")
    job = Job(
        id="job_source",
        status="succeeded",
        spec={"params": {"dataset_id": parent.id, "operation": "prepare_ufo_native_motion"}},
    )

    async def get(cls, key):
        return {(Dataset, parent.id): parent, (Run, run.id): run, (Job, job.id): job}.get((cls, key))

    async def scalar(query):
        return version

    async def scan(_, name):
        path = Path(name)
        return {"files": 1, "samples": [{"sha256": sha256(path), "size": path.stat().st_size}]}

    async def snapshot(_, name):
        return {"path": str(tmp_path / "snapshots/code"), "content_hash": "sha256:" + "d" * 64}

    async def image():
        return "sha256:" + "e" * 64

    settings = SimpleNamespace(
        data_root=tmp_path, pbfm_root=tmp_path / "pbfm", assets_root=tmp_path / "assets"
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: settings)
    monkeypatch.setattr(EbdClient, "scan", scan)
    monkeypatch.setattr(EbdClient, "snapshot", snapshot)
    monkeypatch.setattr(routes_pbfm, "image_digest", image)
    body = routes_pbfm.UfoDataIn(
        name="derived",
        parent_dataset_id=parent.id,
        split_groups={"train": ["walk2"], "validation": ["walk1"], "acceptance": ["walk4"]},
        excluded_groups=["official_rpo_unknown_ancestry"],
    )
    session = SimpleNamespace(get=get, scalar=scalar)
    plan = validate_data_plan(await build_registered_ufo_plan(body, session))
    assert plan["parent_version_id"] == version.id
    rendered = render(spec_for(plan, "ds_derived"), "run_derived")
    assert len([mount for mount in rendered.mounts if mount[1].startswith("/motion-source/")]) == 2
    assert (await build_registered_ufo_plan(body, session))["hash"] == plan["hash"]
    (root / "holdout/validation.pkl").write_bytes(b"mutated after publication")
    with pytest.raises(ValueError, match="changed after validation"):
        await build_registered_ufo_plan(body, session)
