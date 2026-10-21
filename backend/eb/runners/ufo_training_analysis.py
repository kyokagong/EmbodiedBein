"""Read-only native training evidence, executed in the task container."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path


def snapshot_csv(path: Path) -> tuple[bytes, list[dict]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Missing or oversized training evidence: {path.name}")
    data = path.read_bytes()
    # Preserve complete records only when a monitor is still appending.
    complete = data[: data.rfind(b"\n") + 1]
    return complete, list(csv.DictReader(io.StringIO(complete.decode())))


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Native training CSV contains no complete metric rows")
    series = {}
    for key in rows[0]:
        values = [(number(row.get("distributed/optimizer_steps")), number(row.get(key))) for row in rows]
        points = [(x, y) for x, y in values if x is not None and y is not None]
        if not points:
            continue
        window = max(1, len(points) // 4)
        series[key] = {
            "points": len(points),
            "first_update": points[0][0],
            "last_update": points[-1][0],
            "first_window_mean": statistics.fmean(y for _, y in points[:window]),
            "last_window_mean": statistics.fmean(y for _, y in points[-window:]),
            "minimum": min(y for _, y in points),
            "maximum": max(y for _, y in points),
            "window_points": window,
            "missing_or_nonnumeric": len(rows) - len(points),
        }
    if "distributed/optimizer_steps" not in series:
        raise ValueError("Native training CSV has no actual agent-update axis")
    return series


def build_training_evidence(output_root: Path, plan: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    outputs = output_root / "outputs"
    analysis = outputs / "analysis"
    analysis.mkdir(exist_ok=True)
    train_path = outputs / "native/train_log.txt"
    raw, rows = snapshot_csv(train_path)
    summary = summarize(rows)
    raw_snapshot = analysis / "train-log-snapshot.csv"
    raw_snapshot.write_bytes(raw)
    files = []

    def register(path: Path, **metadata):
        files.append(
            {
                "path": path.relative_to(outputs).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **metadata,
            }
        )

    register(raw_snapshot, axis="agent_updates", source="native/train_log.txt")
    selected = [
        key
        for key in summary
        if any(word in key.lower() for word in ("loss", "orth", "actor", "critic", "discriminator"))
    ]
    selected += [key for key in ("FPS", "duration [minutes]", "timestep", "local_timestep") if key in summary]
    if not selected:
        raise ValueError("Native training evidence lacks numerical learning metrics")
    for page in range(0, len(selected), 12):
        keys = selected[page : page + 12]
        figure, axes = plt.subplots(
            (len(keys) + 1) // 2, 2, figsize=(14, 3 * ((len(keys) + 1) // 2)), squeeze=False
        )
        for axis, key in zip(axes.flat, keys):
            points = [(number(row.get("distributed/optimizer_steps")), number(row.get(key))) for row in rows]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            axis.plot([x for x, _ in points], [y for _, y in points], marker=".", linewidth=1)
            axis.set(title=key, xlabel="Native agent updates")
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(keys) :]:
            axis.set_visible(False)
        figure.suptitle(f"{plan['run_id']} | raw native window metrics")
        figure.tight_layout()
        path = analysis / f"training-{page // 12 + 1}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        register(path, axis="agent_updates")

    resource_path = outputs / "resources.csv"
    resource_summary = {"status": "unavailable"}
    if resource_path.is_file():
        resource_raw, resource_rows = snapshot_csv(resource_path)
        resource_snapshot = analysis / "resource-snapshot.csv"
        resource_snapshot.write_bytes(resource_raw)
        register(resource_snapshot, axis="unix_seconds", source="resources.csv", snapshot=True)
        figure, axes = plt.subplots(2, 2, figsize=(14, 8))
        peaks = {}
        for axis, key in zip(
            axes.flat, ("gpu_memory_mib", "gpu_utilization_percent", "cpu_percent", "memory_mib")
        ):
            groups = (
                sorted({row.get("gpu_index", "") for row in resource_rows})
                if key.startswith("gpu_")
                else [""]
            )
            for group in groups:
                selected_rows = [
                    row
                    for row in resource_rows
                    if not key.startswith("gpu_") or row.get("gpu_index") == group
                ]
                points = [(number(row.get("wall_unix_s")), number(row.get(key))) for row in selected_rows]
                points = [(x, y) for x, y in points if x is not None and y is not None]
                if not points:
                    continue
                start = min(x for x, _ in points)
                axis.plot(
                    [x - start for x, _ in points],
                    [y for _, y in points],
                    label=f"GPU {group}" if group else "container",
                )
                peaks[f"{key}/{group or 'container'}"] = max(y for _, y in points)
            axis.set(title=key, xlabel="Seconds from first resource sample")
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend()
        figure.tight_layout()
        path = analysis / "resources.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        register(path, axis="wall_seconds")
        resource_summary = {
            "status": "snapshot",
            "rows": len(resource_rows),
            "peaks": peaks,
            "scope": "GPU assigned-device totals; CPU/RAM container; original continues until exit",
        }
    result = {
        "schema_version": "eb/ufo-native-training-analysis/v1",
        "run_id": plan["run_id"],
        "target_updates": plan["target_updates"],
        "native_metrics": summary,
        "resources": resource_summary,
        "decision": "wait_for_independent_validation",
        "limitations": [
            "Native loss scales and directions differ; no synthetic total loss or qualification.",
            "Tracking inside training updates expert priorities and is not independent validation.",
            "CSV step fields retain the native collect-before-update timing.",
        ],
    }
    path = analysis / "summary.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    register(path)
    for name in ("train_log.txt", "ufo_tracking_eval.csv"):
        path = outputs / "native" / name
        if path.is_file():
            register(
                path, axis="agent_updates" if name == "train_log.txt" else "global_env_steps", original=True
            )
    index = {
        "schema_version": "eb/ufo-native-metrics-index/v1",
        "run_id": plan["run_id"],
        "files": files,
        "decision": "wait_for_independent_validation",
    }
    (outputs / "metrics-index.json").write_text(json.dumps(index, sort_keys=True, indent=2, allow_nan=False))
    return index
