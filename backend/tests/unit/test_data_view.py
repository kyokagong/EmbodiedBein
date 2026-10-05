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
    found = episode_media(tmp_path, "ep_42", [{"episode_id": "ep_42", "streams": [
        {"path": "ep_42_reach.mp4", "from_timestamp": 0, "to_timestamp": 1}]}])
    assert found["missing"] is None
    assert found["path"].endswith("ep_42_reach.mp4")
    assert found["action_series"] == []


def test_v3_overview_is_compact_and_does_not_pass_g2_on_frequency_mismatch():
    from eb.domain.data_view import compact_dataset_stats, dataset_episode_page, dataset_overview
    stats = {"episodes": [{"episode_id": str(i)} for i in range(1693)], "episodes_count": 1693,
             "frames_count": 273465, "fps": 10, "features": {"action": {"shape": [7]},
                                                              "observation.state": {"shape": [8]}}}
    environment = {"action": {"dim": 7}, "observation": {"dim": 8}, "control": {"policy_hz": 20}}
    out = dataset_overview(name="libero", fmt="lerobot", license=None, embodiment_env_id="libero",
                           versions=[], summary_stats=stats, environment=environment)
    assert out["episodes"] == 1693
    assert (out["action_dim"], out["state_dim"], out["fps"]) == (7, 8, 10)
    assert out["g2"] == "不兼容"
    assert "fps: 10 ≠ 20" in out["g2_foot"]
    quality = quality_summary(out["funnel"], license=None, compatible_envs=[],
                              stats={**stats, "interface_compatibility": out["interface_compatibility"]})
    assert not any(item["reason"] == "no_action_dim" for item in quality["items"])
    assert any(item["reason"] == "fps_mismatch" for item in quality["items"])
    assert "episodes" not in compact_dataset_stats(stats)
    page = dataset_episode_page(stats, 50, 50)
    assert len(page["items"]) == 50 and page["next_cursor"] == 100
    assert page["items"][0]["episode_id"] == "50"
    assert dataset_episode_page(stats, 1690, 50)["next_cursor"] is None
    environment["control"]["policy_hz"] = 10
    matching = dataset_overview(name="libero", fmt="lerobot", license=None, embodiment_env_id="libero",
                                versions=[], summary_stats=stats, environment=environment)
    assert matching["g2"] == "待验收"
