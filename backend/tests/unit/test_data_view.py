from pathlib import Path

from eb.domain.data_view import (
    FUNNEL_STAGE_NAMES,
    episode_media,
    license_gate,
    normalize_funnel,
    quality_summary,
    slice_preview,
)

SEED_EGO4D = 155369


def test_empty_funnel_is_zeros_not_seeded_ego4d():
    out = normalize_funnel({})
    assert out["kind"] == "scan"
    assert out["empty"] is True
    assert [s["name"] for s in out["stages"]] == FUNNEL_STAGE_NAMES
    assert all(s["n"] == 0 for s in out["stages"])
    assert SEED_EGO4D not in {s["n"] for s in out["stages"]}
    assert out["sensitivity"] == []


def test_scan_funnel_keeps_real_counts():
    raw = {
        "source": {"n": 17, "rejected": []},
        "raw": {"n": 17, "rejected": [{"reason": "empty_file", "count": 2}]},
        "valid_episode": {"n": 9, "rejected": []},
        "retargeted": {"n": 0, "rejected": []},
        "canonical": {"n": 0, "rejected": []},
        "slice": {"n": 0, "rejected": []},
    }
    out = normalize_funnel(raw)
    assert out["kind"] == "scan"
    assert out["empty"] is False
    assert out["stages"][0]["n"] == 17
    assert out["stages"][2]["n"] == 9
    assert out["rejected"] == [{"reason": "empty_file", "count": 2}]


def test_compile_funnel_kind_is_explicit():
    raw = {"kind": "compile", "source": {"n": 155369}, "valid_episode": {"n": 35}}
    out = normalize_funnel(raw)
    assert out["kind"] == "compile"
    assert out["stages"][0]["n"] == 155369
    assert out["empty"] is False


def test_nc_license_blocks_export():
    g = license_gate("mixed/nc_tainted")
    assert g["export_allowed"] is False
    assert g["share_allowed"] is True
    assert g["gate_status"] == "nc"
    assert license_gate("内部")["export_allowed"] is True


def test_quality_maps_empty_file_to_block():
    funnel = normalize_funnel({"raw": {"n": 17, "rejected": [{"reason": "empty_file", "count": 3}]}})
    q = quality_summary(funnel, license="mixed/nc_tainted", compatible_envs=[], stats={})
    assert q["block"] >= 1
    assert q["license"] == 1
    assert any(i["rule"] == "无零字节文件" for i in q["items"] if i["severity"] == "阻断")


def test_slice_preview_without_episode_index_is_honest():
    prev = slice_preview(stats={"files": 17, "samples": [{"path": "a.jsonl"}]}, query={"task": "reach"})
    assert prev["hits"] is None
    assert prev["warning"]
    assert prev["preview"][0]["path"] == "a.jsonl"
    assert prev["can_submit"] is True


def test_episode_does_not_pick_random_video(tmp_path: Path):
    (tmp_path / "other.mp4").write_bytes(b"fake-mp4")
    out = episode_media(tmp_path, "ep_missing")
    assert out["missing"] == "缺媒体"
    assert out["media_url"] is None
    assert out["action_series"] == []
    hit = tmp_path / "ep_42_reach.mp4"
    hit.write_bytes(b"real")
    found = episode_media(tmp_path, "ep_42")
    assert found["missing"] is None
    assert found["path"].endswith("ep_42_reach.mp4")
    assert found["action_series"] == []
