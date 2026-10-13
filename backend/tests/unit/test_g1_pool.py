import pytest


def test_pool_selection_freezes_takes_before_mapping_and_never_splits_subject_variants():
    from eb.runners.g1_lafan_pool import select_pool
    source={f"{take}_subject{subject}_clip3":{} for take in ("walk1","walk3","walk4","run1") for subject in (1,2)}
    rows=select_pool(source)
    assert len(rows)==8
    assert {r["split"] for r in rows}=={"train","validation","test"}
    for take in ("walk1","walk3","walk4","run1"):
        assert len({r["split"] for r in rows if r["source_group"]==take})==1
    with pytest.raises(ValueError):
        select_pool({"../escape_subject1_clip3":{}})
