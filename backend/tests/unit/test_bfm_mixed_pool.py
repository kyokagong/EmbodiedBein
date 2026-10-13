import copy
import pytest


def mixed_manifest():
    from eb.rpo_bfm import BODY_ORDER,CODEC_HASH,MIXED_POOL_CONVERTER_REVISION
    from eb.rpo_wbc import RPO_LAB_JOINT_ORDER
    clips=[]
    for split,count in (("train",5),("validation",1),("test",1)):
        for i in range(count):
            name=f"g1_{split}_{i}"
            clips.append(dict(id=name,file=name+".npz",sha256="b"*64,fps=50,frames=400,
                split=split,source_group=name,valid_frames=400,
                reference_quality={"continuous_9frame_starts":392,"max_body_position_error_m":.001}))
    return dict(schema_version="eb/rpo-bfm-motion/v1",purpose="behavior_prior",usage="train",
        joint_order=list(RPO_LAB_JOINT_ORDER),body_order=list(BODY_ORDER),codec_sha256=CODEC_HASH,
        asset_sha256="a"*64,converter_revision=MIXED_POOL_CONVERTER_REVISION,
        qualification="local_research_reference",source_use="local_noncommercial_no_redistribution",
        source_pool_sha256="c"*64,selection_sha256="d"*64,official_manifest_sha256="e"*64,
        calibration_verified=True,calibration_applied_to_solver=True,
        source_refs=[dict(uri="https://example.org",revision="f"*40,sha256="e"*64,license="NC")],clips=clips)


def test_mixed_pool_requires_independent_usable_groups_and_research_binding():
    from eb.rpo_bfm import validate_motion_manifest
    m=mixed_manifest()
    result=validate_motion_manifest(m,asset_sha256="a"*64)
    assert result["training_ready"] and not result["controller_evaluation_pass"]
    for mutate in (
        lambda m:m.update(source_use="commercial"),
        lambda m:m.update(selection_sha256=""),
        lambda m:m["clips"][0].update(valid_frames=0),
        lambda m:m["clips"][-1].update(split="train"),
        lambda m:m["clips"][0]["reference_quality"].update(max_body_position_error_m=.1),
    ):
        bad=copy.deepcopy(m);mutate(bad)
        with pytest.raises(ValueError):
            validate_motion_manifest(bad,asset_sha256="a"*64)


def test_formal_budget_rejects_smoke_only_data_and_unbounded_epochs():
    from eb.rpo_bfm import validate_training_budget
    assert validate_training_budget(2,{"smoke_ready":True,"training_ready":False})=="smoke"
    assert validate_training_budget(1000,{"smoke_ready":True,"training_ready":True})=="local_research"
    for steps in (0,1001):
        with pytest.raises(ValueError):validate_training_budget(steps,{"training_ready":True})
    with pytest.raises(ValueError):validate_training_budget(1000,{"smoke_ready":True,"training_ready":False})
