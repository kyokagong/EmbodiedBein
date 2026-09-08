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
    stats = stats or {}
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


def episode_media(root: Path, ep: str) -> dict[str, Any]:
    empty = {
        "id": ep,
        "missing": "缺媒体",
        "media_url": None,
        "path": None,
        "action_series": [],
        "state_series": [],
        "annots": [],
    }
    if not ep or not root.exists():
        return empty
    hits: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".mp4", ".webm", ".mkv"}:
            continue
        if _episode_matches(p, ep):
            hits.append(p)
    if not hits:
        return empty
    video = sorted(hits, key=lambda x: x.as_posix())[0]
    return {
        "id": ep,
        "missing": None,
        "media_url": None,
        "path": str(video),
        "action_series": [],
        "state_series": [],
        "annots": [],
        "note": "无并列关节轨迹时只播视频，不画占位零线",
    }


def _episode_matches(path: Path, ep: str) -> bool:
    stem = path.stem
    posix = path.as_posix()
    if ep.isdigit():
        return stem == ep or stem.endswith(f"_{ep}") or stem.startswith(f"{ep}_")
    return ep in stem or ep in posix


def dataset_overview(
    *,
    name: str,
    fmt: str,
    license: str | None,
    embodiment_env_id: str | None,
    versions: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = versions[-1] if versions else {}
    stats = latest.get("stats") or {}
    funnel = normalize_funnel(latest.get("funnel"))
    lic = license_gate(license)
    files = int(stats.get("files") or funnel["stages"][0]["n"] or 0)
    episodes = int(stats.get("episodes") or funnel["stages"][2]["n"] or 0)
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
    return {
        "layer": layer,
        "files": files,
        "episodes": episodes,
        "bytes": stats.get("bytes"),
        "duration_hist": duration_hist,
        "task_counts": stats.get("task_counts") or {},
        "g2": g2,
        "g2_foot": "无机器人动作维" if g2 == "待定" else "",
        "license_gate": lic,
        "funnel": funnel,
        "content_hash": latest.get("content_hash"),
        "kinds": kinds,
    }
