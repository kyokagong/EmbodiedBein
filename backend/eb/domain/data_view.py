from __future__ import annotations

from pathlib import Path
from typing import Any

FUNNEL_STAGE_NAMES = [
    "source",
    "raw",
    "valid_episode",
    "retargeted",
    "canonical",
    "slice",
]

NC_MARKERS = ("nc", "non-commercial", "noncommercial", "ego4d")


def normalize_funnel(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    kind = raw.get("kind") or "scan"
    stages: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for name in FUNNEL_STAGE_NAMES:
        layer = raw.get(name)
        if not isinstance(layer, dict):
            layer = {"n": int(layer or 0), "rejected": []} if layer else {"n": 0, "rejected": []}
        n = int(layer.get("n") or 0)
        rej = list(layer.get("rejected") or [])
        stages.append({"name": name, "n": n, "rejected": rej})
        rejected.extend(rej)
    empty = all(s["n"] == 0 for s in stages)
    return {
        "kind": kind,
        "empty": empty,
        "stages": stages,
        "rejected": rejected,
        "sensitivity": list(raw.get("sensitivity") or []),
        "has_compile_report": kind == "compile" and not empty,
    }


def license_gate(license_name: str | None) -> dict[str, Any]:
    text = (license_name or "").lower()
    if not text:
        return {
            "gate_status": "unknown",
            "export_allowed": False,
            "share_allowed": False,
            "label": "未标注",
        }
    nc = any(m in text for m in NC_MARKERS)
    if nc:
        return {
            "gate_status": "nc",
            "export_allowed": False,
            "share_allowed": True,
            "label": "NC · 禁止对外分发",
        }
    return {
        "gate_status": "ok",
        "export_allowed": True,
        "share_allowed": True,
        "label": license_name or "内部",
    }


def quality_summary(
    funnel: dict[str, Any],
    *,
    license: str | None,
    compatible_envs: list[Any] | None,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = compact_dataset_stats(stats)
    items: list[dict[str, Any]] = []
    for r in funnel.get("rejected") or []:
        reason = r.get("reason") or ""
        count = int(r.get("count") or 0)
        if reason == "empty_file":
            items.append({"severity": "阻断", "rule": "无零字节文件", "count": count, "reason": reason})
        elif reason in {"frame_mismatch", "video_row_mismatch"}:
            items.append(
                {"severity": "阻断", "rule": "视频帧数 == parquet 行数", "count": count, "reason": reason}
            )
        else:
            items.append({"severity": "警告", "rule": reason, "count": count, "reason": reason})
    if not (compatible_envs or stats.get("action_dim")):
        items.append(
            {
                "severity": "警告",
                "rule": "state/action 维度、fps 与本体环境一致（G2）",
                "count": 1,
                "reason": "no_action_dim",
            }
        )
    compatibility = stats.get("interface_compatibility", {})
    for check in compatibility.get("checks", []):
        if check["status"] == "mismatch":
            items.append({"severity": "阻断", "rule": f"G2 {check['field']}: {check['observed']} ≠ {check['expected']}",
                          "count": 1, "reason": f"{check['field']}_mismatch"})
    if stats.get("action_dim") and compatibility.get("status") != "compatible":
        items.append({"severity": "警告", "rule": "动作语义、时间映射和 loader 尚未验收", "count": 1,
                      "reason": "interface_not_verified"})
    lic = license_gate(license)
    license_n = 0
    if lic["gate_status"] != "ok":
        license_n = 1
        items.append(
            {
                "severity": "许可",
                "rule": "许可标签存在且传染性已推导",
                "count": 1,
                "reason": lic["gate_status"],
            }
        )
    block = sum(1 for i in items if i["severity"] == "阻断")
    warn = sum(1 for i in items if i["severity"] == "警告")
    hint = sum(1 for i in items if i["severity"] == "提示")
    return {
        "block": block,
        "warn": warn,
        "hint": hint,
        "license": license_n,
        "items": items,
        "license_gate": lic,
    }


def slice_preview(stats: dict[str, Any] | None, query: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    query = query or {}
    samples = list(stats.get("samples") or [])
    episodes = stats.get("episodes")
    if isinstance(episodes, list) and episodes:
        hits = episodes
        task = query.get("task")
        if task:
            hits = [e for e in hits if (e.get("task") if isinstance(e, dict) else None) == task]
        return {
            "hits": len(hits),
            "total": len(episodes),
            "preview": hits[:12],
            "warning": "",
            "can_submit": True,
            "query": query,
        }
    return {
        "hits": None,
        "total": stats.get("files") or 0,
        "preview": samples[:12],
        "warning": "缺少 episode 索引，命中数为空——不会用占位 1128 冒充预览",
        "can_submit": True,
        "query": query,
    }


def episode_media(root: Path, ep: str, episodes: list[dict] | None = None) -> dict[str, Any]:
    payload = {"id": ep, "missing": "缺媒体", "media_url": None, "path": None,
               "action_series": [], "state_series": [], "annots": [], "streams": []}
    episode = next((row for row in episodes or [] if str(row.get("episode_id")) == ep), None)
    if not episode:
        return payload
    streams = episode.get("streams", [])
    if not streams:
        return payload
    video = (root / streams[0]["path"]).resolve()
    if not video.is_relative_to(root.resolve()) or not video.is_file():
        return payload
    return {**payload, "missing": None, "path": str(video), "streams": streams,
            "from_timestamp": streams[0]["from_timestamp"], "to_timestamp": streams[0]["to_timestamp"]}


def dataset_overview(
    *,
    name: str,
    fmt: str,
    license: str | None,
    embodiment_env_id: str | None,
    versions: list[dict[str, Any]],
    summary_stats: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = versions[-1] if versions else {}
    stats = compact_dataset_stats({**(latest.get("stats") or {}), **(summary_stats or {})})
    funnel = normalize_funnel(latest.get("funnel"))
    lic = license_gate(license)
    files = int(stats.get("files") or funnel["stages"][0]["n"] or 0)
    episodes = int(stats.get("episodes_count") or stats.get("episodes") or funnel["stages"][2]["n"] or 0)
    kinds = stats.get("kinds") or {}
    if not episodes and isinstance(kinds, dict):
        episodes = int(kinds.get("ego_jsonl") or 0)
    duration_hist = stats.get("duration_hist") or []
    layer = (latest.get("meta") or {}).get("layer") or stats.get("layer") or "L1"
    g2 = "待定"
    if latest.get("compatible_envs"):
        g2 = "通过"
    elif files and not latest.get("compatible_envs"):
        g2 = "待定"
    interface = dataset_interface_check(stats, environment)
    if summary_stats and summary_stats.get("features"):
        g2 = "不兼容" if interface["status"] == "incompatible" else "待验收"
    return {
        "layer": layer,
        "files": files,
        "episodes": episodes,
        "bytes": stats.get("bytes"),
        "duration_hist": duration_hist,
        "task_counts": stats.get("task_counts") or {},
        "g2": g2,
        "g2_foot": interface["reason"] if stats.get("features") else "无机器人动作维" if g2 == "待定" else "",
        "interface_compatibility": interface,
        "fps": stats.get("fps"),
        "action_dim": stats.get("action_dim"),
        "state_dim": stats.get("state_dim"),
        "frames": stats.get("frames_count"),
        "license_gate": lic,
        "funnel": funnel,
        "content_hash": latest.get("content_hash"),
        "kinds": kinds,
    }


def compact_dataset_stats(stats: dict | None) -> dict:
    stats = stats or {}
    result = {key: value for key, value in stats.items() if key != "episodes"}
    episodes = stats.get("episodes")
    if isinstance(episodes, list):
        result.setdefault("episodes_count", len(episodes))
    elif isinstance(episodes, int):
        result.setdefault("episodes_count", episodes)
    features = stats.get("features", {})
    for feature, output in (("action", "action_dim"), ("observation.state", "state_dim")):
        shape = features.get(feature, {}).get("shape", [])
        if len(shape) == 1:
            result[output] = shape[0]
    return result


def dataset_interface_check(stats: dict, environment: dict | None) -> dict:
    observed = compact_dataset_stats(stats)
    expected = environment or {}
    checks = []
    for field, required in (("action_dim", expected.get("action", {}).get("dim")),
                            ("state_dim", expected.get("observation", {}).get("dim")),
                            ("fps", expected.get("control", {}).get("policy_hz"))):
        actual = observed.get(field)
        checks.append({"field": field, "observed": actual, "expected": required,
                       "status": "unknown" if actual is None or required is None else
                                 "match" if actual == required else "mismatch"})
    mismatches = [check for check in checks if check["status"] == "mismatch"]
    return {"status": "incompatible" if mismatches else "not_verified", "checks": checks,
            "reason": "; ".join(f"{c['field']}: {c['observed']} ≠ {c['expected']}" for c in mismatches)
            or "动作语义、时序映射与消费端 loader 尚未验收"}


def dataset_episode_page(stats: dict | None, cursor: int, limit: int) -> dict:
    episodes = (stats or {}).get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []
    end = min(cursor + limit, len(episodes))
    return {"items": episodes[cursor:end], "total": len(episodes),
            "next_cursor": end if end < len(episodes) else None}
