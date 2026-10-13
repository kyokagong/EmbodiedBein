import numpy as np
import pytest


def test_calibration_offsets_close_position_and_rotation_independently():
    from eb.runners.g1_rpo_calibration import check_offsets
    source = {"hips":np.eye(4), "left_wrist":np.eye(4)}
    target = {k:v.copy() for k,v in source.items()}
    source["left_wrist"][:3,3] = [1,0,0]
    target["left_wrist"][:3,3] = [2,.2,0]
    scales = {"hips":1., "left_wrist":2.}
    offsets = {"hips":((0,0,0),(0,0,0,1)), "left_wrist":((0,.2,0),(0,0,0,1))}
    assert check_offsets(source,target,scales,offsets)["max_position_error_m"] < 1e-8
    with pytest.raises(ValueError):
        check_offsets(source,target,scales,{**offsets,"left_wrist":((0,0,0),(0,0,0,1))})
    with pytest.raises(ValueError):
        check_offsets(source,target,scales,{**offsets,"left_wrist":((0,.2,0),(0,0,1,0))})
    with pytest.raises(ValueError):
        check_offsets(source,target,{**scales,"left_wrist":float("nan")},offsets)


def test_g1_codec_input_requires_bound_calibration_and_preserves_mask(tmp_path):
    import hashlib,json
    from eb.runners.rpo_bfm_motion import load_g1_reference
    from eb.rpo_bfm import RPO_LAB_JOINT_ORDER,BODY_ORDER
    path=tmp_path/"rpo_g1_walk.npz"
    np.savez(path,q=np.zeros((100,23),np.float32),root_pos=np.tile([0,0,.8],(100,1)),
        root_quat=np.tile([1,0,0,0],(100,1)),fps=50,valid=np.r_[np.zeros(10,bool),np.ones(90,bool)],
        joint_names=np.asarray(RPO_LAB_JOINT_ORDER),body_names=np.asarray(BODY_ORDER),timestamps=np.arange(100)/50)
    cal={"verified":True,"method":"hhtools_constrained_chain_alignment_soma_offsets_v1",
         "asset_sha256":"a"*64,"source_export_sha256":"b"*64}
    cal_path=tmp_path/"r2r-calibration.json"
    cal_path.write_text(json.dumps(cal))
    receipt={"schema_version":"eb/rpo-g1-r2r-trial/v1","status":"succeeded","frames":100,"fps":50,
        "asset_sha256":"a"*64,"calibration_verified":True,"calibration_applied_to_solver":True,"calibration_sha256":hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "source_export_sha256":"b"*64,"source_use":"local_noncommercial_no_redistribution",
        "trajectory_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_group":"walk1_subject1",
        "source_refs":[],"source_clip_id":"walk1_subject1_clip3"}
    rpath=tmp_path/"retarget-receipt.json"
    rpath.write_text(json.dumps(receipt))
    raw,metadata=load_g1_reference(tmp_path,"a"*64)
    assert raw["valid"].sum()==90 and raw["dof_pos"].shape==(100,23)
    assert metadata["source_group"]=="walk1_subject1"
    for patch in ({"calibration_verified":False},{"calibration_applied_to_solver":False},{"asset_sha256":"c"*64},{"calibration_sha256":"0"*64}):
        rpath.write_text(json.dumps({**receipt,**patch}))
        with pytest.raises(ValueError):
            load_g1_reference(tmp_path,"a"*64)


def test_g1_codec_adapter_binds_published_source_and_rejects_network(tmp_path,monkeypatch):
    import json,hashlib
    from pathlib import Path
    from types import SimpleNamespace
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobKind,JobSpec
    from eb.rpo_bfm import G1_IMAGE
    source=tmp_path/"datasets/ds_abcdef/content"
    source.mkdir(parents=True)
    receipt=source/"retarget-receipt.json"
    receipt.write_text(json.dumps({"calibration_verified":True}))
    monkeypatch.setattr("eb.settings.get_settings",lambda:SimpleNamespace(datasets_root=tmp_path/"datasets",data_root=tmp_path/"state",assets_root=Path("/data/kk/assets")))
    spec=JobSpec(kind=JobKind.DATA_CONVERT,embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"operation":"convert_rpo_bfm_g1","dataset_id":"ds_123abc","source_dataset_id":"ds_abcdef",
                "source_retarget_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()},
        resources={"gpus":1,"memory_gb":24,"shm_gb":8},runtime={"network":"none"})
    rendered=DataConvertAdapter().render(spec,"run_test")
    assert rendered.image==G1_IMAGE and "g1_reference" in rendered.command
    assert next(m for m in rendered.mounts if m[1]=="/workspace/motions")== (str(source),"/workspace/motions","ro")
    spec.runtime["network"]="bridge"
    with pytest.raises(ValueError):
        DataConvertAdapter().render(spec,"run_test")
