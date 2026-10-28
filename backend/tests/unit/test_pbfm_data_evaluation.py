from copy import deepcopy

import pytest

from eb.pbfm_data import ENV_ID, require_pbfm_dataset, validate_source_manifest
from eb.pbfm_evaluation import build_eval_plan, route_outcome, validate_case_rows, validate_eval_plan


def source():
    return {
        "schema_version": "eb/pbfm-motion-source/v1",
        "robot_name": "rpo",
        "joint_order": [f"j{x}" for x in range(23)],
        "coordinate_system": "z_up",
        "root_quat_order": "xyzw",
        "dof_unit": "rad",
        "splits": {
            s: [
                {
                    "path": s + ".pkl",
                    "format": "ufo_pickle",
                    "sha256": c * 64,
                    "ancestor_ids": [s],
                    "bytes": 100,
                }
            ]
            for s, c in zip(("train", "validation", "acceptance"), "abc")
        },
    }


def test_source_requires_real_disjoint_ancestry_and_rpo():
    validate_source_manifest(source())
    for field, value in (("robot_name", "g1"), ("joint_order", ["j"] * 23)):
        item = source()
        item[field] = value
        with pytest.raises(ValueError):
            validate_source_manifest(item)
    item = source()
    item["splits"]["acceptance"][0]["ancestor_ids"] = ["train"]
    with pytest.raises(ValueError, match="ancestr"):
        validate_source_manifest(item)
    item = source()
    item["splits"]["acceptance"] = []
    with pytest.raises(ValueError):
        validate_source_manifest(item)


def test_source_rejects_escape_links_and_unfrozen_data():
    for field, value in (
        ("path", "../train.pkl"),
        ("path", "/train.pkl"),
        ("sha256", "missing"),
        ("ancestor_ids", []),
    ):
        item = source()
        item["splits"]["train"][0][field] = value
        with pytest.raises(ValueError):
            validate_source_manifest(item)


def plan():
    return build_eval_plan(
        model_version_id="mdl_test",
        model_hash="a" * 64,
        export_hash="b" * 64,
        dataset_manifest="c" * 64,
        image_digest="sha256:" + "d" * 64,
        source_refs={"pbfm": {"path": "/snapshot", "snapshot_hash": "sha256:" + "e" * 64}},
        camera_contract={"body": "torso_link"},
        robot_geometry={
            "robot": "rpo",
            "foot_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "xml_sha256": "f" * 64,
        },
    )


def test_frozen_suite_exact_cases_and_hash():
    value = plan()
    validate_eval_plan(value)
    assert len(value["cases"]) == 16
    broken = deepcopy(value)
    broken["cases"].pop()
    with pytest.raises(ValueError):
        validate_eval_plan(broken)
    broken = deepcopy(value)
    broken["robot_geometry"]["robot"] = "g1"
    with pytest.raises(ValueError):
        validate_eval_plan(broken)


def _action_render_spec(tmp_path, monkeypatch, operation):
    from types import SimpleNamespace

    from eb.pbfm_evaluation import action_spec
    from eb.pbfm_io import fingerprint

    monkeypatch.setattr("eb.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    value = plan()
    if operation == "export_pbfm_policy":
        value = {
            key: value[key] for key in ("model_version_id", "model_hash", "dataset_manifest", "image_digest")
        }
        value.update(
            schema_version="eb/pbfm-export-plan/v1", operation=operation,
            prompt_manifest_sha256="a" * 64, prompt_file_sha256="b" * 64, motion_id=0,
        )
    value["source_refs"] = {}
    for name, letter in (("pbfm", "e"), ("asset", "f")):
        root = tmp_path / "snapshots" / (letter * 64) / "code"
        root.mkdir(parents=True)
        value["source_refs"][name] = {
            "kind": "content_snapshot", "path": str(root), "snapshot_hash": "sha256:" + letter * 64,
            "manifest_path": str(root.parent / "manifest.json"),
        }
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"frozen model")
    value.update(
        model_path=str(model), model_files=[{"path": "model.safetensors"}],
        prompt_path=str(tmp_path / "holdout"), export_path=str(tmp_path / "export"),
    )
    value["hash"] = fingerprint({key: item for key, item in value.items() if key != "hash"})
    return action_spec(value)


@pytest.mark.parametrize(
    ("operation", "runner"),
    [
        ("export_pbfm_policy", "eb.runners.pbfm_export"),
        ("evaluate_pbfm_terrain", "eb.runners.pbfm_eval"),
    ],
)
def test_pbfm_actions_render_request_egl_driver_capabilities(tmp_path, monkeypatch, operation, runner):
    from eb.adapters import get_adapter

    spec = _action_render_spec(tmp_path, monkeypatch, operation)
    rendered = get_adapter(spec).render(spec, "run_preview")
    assert rendered.command[2] == runner
    assert rendered.env.get("NVIDIA_DRIVER_CAPABILITIES") == "compute,utility,graphics"


@pytest.mark.parametrize("operation", ["export_pbfm_policy", "evaluate_pbfm_terrain"])
def test_pbfm_actions_render_persisted_empty_code_refs(tmp_path, monkeypatch, operation):
    from eb.adapters import get_adapter
    from eb.domain.job import CodeRefs, JobSpec

    spec = _action_render_spec(tmp_path, monkeypatch, operation)
    # Orchestrator.submit materializes CodeRefs() before JSON persistence.
    stored = spec.model_copy(update={"code_refs": CodeRefs()}).model_dump_json()
    persisted = JobSpec.model_validate_json(stored)
    assert get_adapter(persisted).render(persisted, "run_queued").command == (
        get_adapter(spec).render(spec, "run_preview").command
    )


@pytest.mark.parametrize("operation", ["export_pbfm_policy", "evaluate_pbfm_terrain"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("code_refs", "embodiment"), {"path": "/unfrozen/embodiment"}),
        (("code_refs", "stack"), {"path": "/unfrozen/stack"}),
        (("resources", "gpus"), 2),
        (("runtime", "network"), "bridge"),
        (("image",), "sha256:" + "1" * 64),
        (("inputs", "base_model"), "/unfrozen/model"),
        (("params", "pbfm_plan", "model_hash"), "1" * 64),
    ],
)
def test_pbfm_actions_reject_persisted_spec_tampering(tmp_path, monkeypatch, operation, path, value):
    from eb.adapters import get_adapter
    from eb.domain.job import CodeRefs, JobSpec

    stored = _action_render_spec(tmp_path, monkeypatch, operation).model_dump(mode="json")
    stored["code_refs"] = CodeRefs().model_dump(mode="json")
    target = stored
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    spec = JobSpec.model_validate(stored)
    with pytest.raises(ValueError):
        get_adapter(spec).render(spec, "run_tampered")


def test_standing_survival_does_not_pass_route_and_direction_comes_from_heights():
    assert not route_outcome([[0, 0, 1], [0.1, 0, 1]], [0, 0], alive=True, constraints_ok=True)["success"]
    result = route_outcome([[0, 0, 1], [2.3, 0, 1.2]], [0, 0.2], alive=True, constraints_ok=True)
    assert result["success"] and result["vertical_direction"] == "ascent"
    assert not route_outcome([[0, 0, 1], [2.3, 0, 1.2]], [0, 0.2], alive=False, constraints_ok=True)[
        "success"
    ]


def test_incomplete_cases_and_media_are_rejected():
    with pytest.raises(ValueError):
        validate_case_rows(plan(), [])
    rows = valid_rows()
    rows[1]["video"] = rows[0]["video"]
    with pytest.raises(ValueError, match="unique"):
        validate_case_rows(plan(), rows)


def valid_rows():
    return [
        {
            "case_id": case["id"],
            "seed": case["seed"],
            "family": case["family"],
            "commands": case["commands"],
            "diagnostics": {
                "joint_tracking_rmse_rad": 0.1,
                "body_tracking_rmse_m": 0.02,
                "velocity_rmse_m_s": 0.05,
                "foot_slip_m_s": 0.01,
                "stance_samples": 100,
                "foot_slip_status": "measured",
                "max_contact_force_n": 100.0,
                "min_toe_clearance_m": -0.001,
                "min_heel_clearance_m": 0.002,
                "action_saturation_fraction": 0.001,
            },
            "condition": condition,
            "video": f"videos/{case['id']}-{condition}.mp4",
            "depth": f"depth/{case['id']}-{condition}.npz",
            "depth_video": f"depth-videos/{case['id']}-{condition}.mp4",
            "depth_video_verified": True,
            "depth_video_frames": 1000,
            "depth_video_fps": 50,
            "trace": f"traces/{case['id']}-{condition}.npz",
            "complete": True,
            "success": True,
            "video_verified": True,
            "depth_verified": True,
            "video_fps": 25,
            "video_frames": 501,
            "depth_frames": 1000,
            "steps": 1000,
            "seconds": 20,
            "terminated": False,
            "goal_reached": not case["regression"],
            "constraints_ok": True,
            "forbidden_contact_count": 0,
            "forbidden_contact_peak_n": 0.0,
            "max_tilt_rad": 0.1,
            "min_root_clearance_m": 0.7,
            "max_torque_ratio": 0.8,
            "max_joint_excess_rad": 0.0,
            "max_step_displacement_m": 0.01,
            "max_drift_m": 0.05 if case["regression"] else 2.3,
            "final_speed_m_s": 0.01,
            "route": {
                "progress_m": 0.01 if case["regression"] else 2.3,
                "max_lateral_m": 0.01,
                "ground_height_delta_m": 0.0,
                "success": True,
            },
        }
        for case in plan()["cases"]
        for condition in ("normal", "occluded")
    ]


def test_success_is_recomputed_and_complete_media_does_not_promote_acceptance():
    result = validate_case_rows(plan(), valid_rows())
    assert result["perception_rollouts"] == 32
    assert result["media_categories_absent"] == ["failure"]
    rows = valid_rows()
    rows[0]["route"]["progress_m"] = 0
    with pytest.raises(ValueError, match="standing"):
        validate_case_rows(plan(), rows)
    rows = valid_rows()
    rows[0]["video_verified"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_case_rows(plan(), rows)
    assert plan()["suite_spec"]["formal_acceptance"] is False
    rows = valid_rows()
    rows[0]["forbidden_contact_count"] = 1
    rows[0]["forbidden_contact_peak_n"] = 10.0
    with pytest.raises(ValueError, match="constraints"):
        validate_case_rows(plan(), rows)


def test_real_dataset_orm_flows_through_api_and_training_binding(tmp_path):
    from eb.api.routes_data import _ds_out
    from eb.pbfm_io import sha256, write_json
    from eb.store.models import Dataset

    (tmp_path / "train.pkl").write_bytes(b"frozen native motion")
    manifest = {
        "schema_version": "eb/pbfm-rpo-motion/v1",
        "dataset_id": "ds_test",
        "split": "train",
        "train_file": "train.pkl",
        "file_sha256": sha256(tmp_path / "train.pkl"),
        "native_loader": {"all_motions_loaded": True},
        "fk": {"verified": True},
        "joint_order": [f"j{x}" for x in range(23)],
    }
    write_json(tmp_path / "manifest.json", manifest)
    validation = {
        "scope": "pbfm_rpo_motion_v1",
        "status": "validated",
        "training_ready": True,
        "manifest_sha256": sha256(tmp_path / "manifest.json"),
        "file_sha256": manifest["file_sha256"],
    }
    dataset = Dataset(
        id="ds_test",
        name="PBFM data",
        source_kind="pbfm_rpo_motion",
        format="pbfm_rpo_motion",
        embodiment_env_id=ENV_ID,
        status="validated",
        dest_path=str(tmp_path),
        summary_stats={"validation": validation},
        meta={"usage": "train"},
    )
    output = _ds_out(dataset)
    assert output["source_kind"] == "pbfm_rpo_motion"
    assert output["validation"]["training_ready"] is True
    assert require_pbfm_dataset(dataset)["storage_ref"] == str(tmp_path)
    (tmp_path / "holdout").mkdir()
    with pytest.raises(ValueError, match="train split"):
        require_pbfm_dataset(dataset)


@pytest.mark.asyncio
async def test_collector_hashes_large_outputs_through_ebd(tmp_path, monkeypatch):
    from eb.pbfm_evaluation import verify_output_index
    from eb.store.ebd import EbdClient

    (tmp_path / "video.mp4").write_bytes(b"opaque compressed data")
    calls = []

    async def scan(_, path):
        calls.append(path)
        return {"files": 1, "samples": [{"path": path, "size": 22, "sha256": "a" * 64}]}

    monkeypatch.setattr(EbdClient, "scan", scan)
    row = {"path": "video.mp4", "bytes": 22, "sha256": "a" * 64}
    assert (await verify_output_index(tmp_path, [row], {"video.mp4"}))["video.mp4"] == row
    assert calls == [str(tmp_path / "video.mp4")]
    with pytest.raises(ValueError, match="integrity"):
        await verify_output_index(tmp_path, [{**row, "sha256": "b" * 64}], {"video.mp4"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["policy/source/pbfm/__init__.py", "policy/source/pbfm/envs/mjlab/__init__.py"]
)
async def test_collector_accepts_scanned_empty_package_markers(tmp_path, monkeypatch, name):
    import hashlib
    from pathlib import Path

    from eb.pbfm_evaluation import verify_output_index
    from eb.store.ebd import EbdClient

    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    calls = []

    async def scan(_, filename):
        calls.append(filename)
        content = Path(filename).read_bytes()
        return {
            "files": 1,
            "samples": [{"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}],
        }

    monkeypatch.setattr(EbdClient, "scan", scan)
    row = {"path": name, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    assert await verify_output_index(tmp_path, [row], {name}) == {name: row}
    assert calls == [str(path)]
    with pytest.raises(ValueError, match="unique"):
        await verify_output_index(tmp_path, [row, row], {name})
    with pytest.raises(ValueError, match="incomplete"):
        await verify_output_index(tmp_path, [row], {"policy/policy.onnx"})
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        await verify_output_index(tmp_path, [row], {name})
    path.unlink()
    target = tmp_path / "empty.py"
    target.write_bytes(b"")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="linked"):
        await verify_output_index(tmp_path, [row], {name})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"path": "policy/policy.onnx"},
        {"path": "video.mp4"},
        {"path": "policy/manifest.json"},
        {"path": "policy/source/pbfm/model.py"},
        {"path": "policy/source/pbfm_other/__init__.py"},
        {"path": "source/pbfm/__init__.py"},
        {"path": "policy/source/pbfm/envs/../__init__.py"},
        {"path": "/policy/source/pbfm/__init__.py"},
        {"sha256": "a" * 64},
        {"bytes": -1},
        {"bytes": False},
    ],
)
async def test_collector_rejects_other_empty_or_tampered_entries(tmp_path, monkeypatch, change):
    import hashlib

    from eb.pbfm_evaluation import verify_output_index
    from eb.store.ebd import EbdClient

    async def scan(*args):
        raise AssertionError("invalid empty entries must fail before scanning")

    monkeypatch.setattr(EbdClient, "scan", scan)
    row = {"path": "policy/source/pbfm/__init__.py", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    row.update(change)
    with pytest.raises(ValueError):
        await verify_output_index(tmp_path, [row], set())


def test_native_vector_reset_preserves_failure_successor():
    from types import SimpleNamespace

    import numpy as np

    from eb.runners.pbfm_eval import advance_with_successor

    core = SimpleNamespace(
        robot_root_states=np.zeros((1, 13)),
        torques=np.zeros((1, 23)),
        dof_pos=np.zeros((1, 23)),
        projected_gravity=np.asarray([[0, 0, -1.0]]),
    )

    def step(action):
        core.robot_root_states[0, 0] = 1.7
        core.robot_root_states[0, 2] = 0.2
        core.torques[:] = 9

    core.step = step
    wrapped = SimpleNamespace(_env=core)

    def qpos(to_numpy=False):
        positions = np.zeros((1, 30))
        positions[0, :3] = core.robot_root_states[0, :3]
        return positions, np.zeros((1, 29))

    wrapped._get_qpos_qvel = qpos

    def vector_step(action, to_numpy=False):
        core.step(action)
        final, velocity = qpos()
        core.robot_root_states[:] = 0
        core.torques[:] = 0
        return {}, 0, np.asarray([True]), np.asarray([False]), {"final_qpos": final, "final_qvel": velocity}

    wrapped.step = vector_step
    _, successor = advance_with_successor(wrapped, np.zeros((1, 23)))
    assert successor["qpos"][0] == 1.7
    assert successor["root"][2] == 0.2
    assert successor["torque"][0] == 9
    assert core.robot_root_states[0, 0] == 0
    assert core.step is step


@pytest.mark.parametrize(
    ("regression", "progress", "expected_goal"),
    [(False, 0.05, False), (False, 2.3, True), (True, 0.05, False), (True, 2.3, False)],
)
def test_completed_case_journal_is_bound_to_model_plan_and_media(
    tmp_path, monkeypatch, regression, progress, expected_goal,
):
    import ast
    import inspect
    import json

    import numpy as np

    from eb.pbfm_io import sha256
    from eb.runners.pbfm_eval import append_completed_case, evaluate_case

    row = {"case_id": "flat-71000", "condition": "normal"}
    # Exercise the production assignment with the actual successor scalar type,
    # followed by the real journal writer; no simulator or JSON fallback.
    assignment = next(node for node in ast.walk(ast.parse(inspect.getsource(evaluate_case)))
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == "goal_reached")
    row["goal_reached"] = eval(compile(ast.Expression(assignment.value), "evaluate_case", "eval"), {
        "case": {"regression": regression}, "root": np.asarray([progress, 0, 0.75], np.float32),
        "roots": [np.asarray([0, 0, 0.75], np.float32)],
        "plan": {"suite_spec": {"route_distance_m": 2.2}},
    })
    for key, suffix in (("video", ".mp4"), ("depth_video", ".mp4"), ("depth", ".npz"), ("trace", ".npz")):
        name = key + suffix
        (tmp_path / name).write_bytes(b"actual indexed test media")
        row[key] = name
    monkeypatch.setenv("EB_RUN_ID", "run_test")
    binding = {"hash": "a" * 64, "model_version_id": "mv_test", "model_hash": "b" * 64}
    append_completed_case(tmp_path, binding, row)
    saved = json.loads((tmp_path / "cases.jsonl").read_text())
    assert type(row["goal_reached"]) is bool and saved["goal_reached"] is expected_goal
    assert saved["run_id"] == "run_test" and saved["plan_hash"] == binding["hash"]
    assert saved["model_hash"] == binding["model_hash"]
    assert len(saved["evidence_files"]) == 4
    assert all(item["sha256"] == sha256(tmp_path / item["path"]) for item in saved["evidence_files"])
