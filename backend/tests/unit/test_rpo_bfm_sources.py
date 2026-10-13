import pytest


def test_100style_probe_acquires_only_fixed_neutral_files_and_rejects_html(tmp_path):
    from eb.runners.rpo_100style_probe import FILES, check_download

    assert set(FILES) == {
        "Neutral_ID.bvh",
        "Neutral_FW.bvh",
        "Neutral_BW.bvh",
        "Neutral_SW.bvh",
        "Frame_Cuts.csv",
    }
    assert all(url.startswith("https://drive.usercontent.google.com/download?") for url in FILES.values())
    with pytest.raises(ValueError):
        check_download("Neutral_FW.bvh", b"<html>sign in</html>")
    with pytest.raises(ValueError):
        check_download("Neutral_FW.bvh", b"HIERARCHY\nROOT Hips { }\nMOTION\nFrames: 0\nFrame Time: 1")
    truncated = (b"HIERARCHY\nROOT Hips { CHANNELS 6 Xposition Yposition Zposition "
                 b"Yrotation Xrotation Zrotation }\nMOTION\nFrames: 9\nFrame Time: 0.016667\n0 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="complete"):
        check_download("Neutral_FW.bvh", truncated)


def test_100style_acquisition_adapter_uses_bounded_network_job(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobKind, JobSpec

    monkeypatch.setattr(
        "eb.settings.get_settings",
        lambda: SimpleNamespace(data_root=tmp_path, hf_endpoint="https://huggingface.co"),
    )
    spec = JobSpec(
        kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"operation": "acquire_100style_probe"},
        runtime={"network": "bridge"},
        resources={"gpus": 0, "memory_gb": 4, "cpus": 2, "shm_gb": 1},
    )
    rendered = ToolsAdapter().render(spec, "run_style")
    assert rendered.command[1].endswith("/rpo_100style_probe.py")
    assert rendered.gpus == 0 and len(rendered.mounts) == 1
    assert rendered.mounts[0][2] == "rw"


def test_xsens_parser_respects_cm_yup_and_intrinsic_channel_order():
    import numpy as np

    from eb.runners.rpo_100style_retarget import parse_bvh

    text = """HIERARCHY
ROOT Hips {
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
JOINT LeftAnkle {
OFFSET 100 -100 0
CHANNELS 3 Yrotation Xrotation Zrotation
End Site { OFFSET 0 0 10 }
}}
MOTION
Frames: 2
Frame Time: 0.0166667
0 100 0 0 0 0 0 0 0
0 100 0 90 0 0 0 0 0
"""
    names, positions, rotations, fps = parse_bvh(text)
    assert names == ["Hips", "LeftAnkle"] and abs(fps - 60) < 0.001
    assert np.allclose(positions[0, 1], [0, 1, 0])  # Y-up/cm → Z-up/metres.
    assert np.allclose(positions[1, 1], [-1, 0, 0], atol=1e-6)
    assert np.allclose(rotations[0, 0], np.eye(3))
    with pytest.raises(ValueError):
        parse_bvh(text.replace("0 100 0 90 0 0 0 0 0", "nan 100 0 90 0 0 0 0 0"))


def test_100style_retarget_adapter_binds_readonly_source_receipt_and_asset(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobKind, JobSpec

    source = tmp_path / "runs/run_abcdef/outputs/100style"
    source.mkdir(parents=True)
    receipt = source / "source-manifest.json"
    receipt.write_text("{}")
    sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "eb.settings.get_settings",
        lambda: SimpleNamespace(data_root=tmp_path, assets_root=Path("/data/kk/assets")),
    )
    spec = JobSpec(
        kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={
            "operation": "retarget_100style_probe",
            "source_run_id": "run_abcdef",
            "source_manifest_sha256": sha,
        },
        runtime={"network": "none"},
        resources={"gpus": 0, "memory_gb": 8, "shm_gb": 1},
    )
    rendered = ToolsAdapter().render(spec, "run_retarget")
    assert (str(source), "/workspace/source", "ro") in rendered.mounts
    assert rendered.command[1].endswith("/rpo_100style_retarget.py")
    spec.params["source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ToolsAdapter().render(spec, "run_invalid")


def test_retarget_calibration_preserves_robot_limb_lengths_and_ground_contact():
    import numpy as np

    from eb.runners.rpo_100style_retarget import BODY_MAP, limb_targets

    names = ["Hips"]
    coordinates = [[0, 0, 1.3]]
    chains = {}
    for side, sign in (("Left", 1), ("Right", -1)):
        names.extend([side + part for part in ("Hip", "Knee", "Ankle", "Shoulder", "Elbow", "Wrist")])
        coordinates.extend(
            [
                [0, sign * 0.1, 1.2],
                [0, sign * 0.1, 0.6],
                [0, sign * 0.1, 0],
                [0, sign * 0.2, 1.7],
                [0, sign * 0.2, 1.2],
                [0, sign * 0.2, 0.9],
            ]
        )
        chains[side] = {
            "hip": np.array([0, sign * 0.1, -0.1]),
            "shoulder": np.array([0, sign * 0.2, 0.2]),
            "leg_lengths": (0.4, 0.4),
            "arm_lengths": (0.3, 0.2),
        }
    targets, root = limb_targets(names, np.array(coordinates)[None], np.eye(3)[None], chains, 2 / 3)
    assert np.isclose(root[0, 2], 0.94)
    ids = {name: i for i, name in enumerate(BODY_MAP)}
    for side in chains:
        assert np.isclose(targets[0, ids[side + "Ankle"], 2], 0.04)
        assert np.isclose(
            np.linalg.norm(targets[0, ids[side + "Knee"]] - targets[0, ids[side + "Ankle"]]), 0.4
        )
        assert np.isclose(
            np.linalg.norm(targets[0, ids[side + "Shoulder"]] - targets[0, ids[side + "Elbow"]]), 0.3
        )
        assert np.isclose(
            np.linalg.norm(targets[0, ids[side + "Elbow"]] - targets[0, ids[side + "Wrist"]]), 0.2
        )
