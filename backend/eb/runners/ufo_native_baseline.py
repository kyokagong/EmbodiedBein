"""Run the real UFO launcher once, then compare its frozen diagnostic output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eb.ufo_native_baseline import BASE_IMAGE, RECEIPT_SCHEMA, RULES, SCHEMA, SOURCE_RUN


def optimizer_layout(optimizers: dict) -> dict:
    """State schema, group membership and actual counters, without tensor copies."""
    result = {}
    for name, optimizer in optimizers.items():
        states = optimizer["state"]
        steps = [float(value["step"]) for value in states.values() if "step" in value]
        if not steps or any(value != 112 for value in steps):
            raise ValueError("Native optimizer internal step did not reach 112")
        result[name] = {
            "entries": len(states),
            "step_minimum": min(steps),
            "step_maximum": max(steps),
            "states": {
                str(index): {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if hasattr(value, "shape")
                    else {"type": type(value).__name__}
                    for key, value in state.items()
                }
                for index, state in states.items()
            },
            "groups": [
                {key: value for key, value in group.items() if key != "params"}
                | {"params": list(group["params"])}
                for group in optimizer["param_groups"]
            ],
        }
    if len(result) != 5:
        raise ValueError("Native FB baseline requires all five optimizers")
    return result


def run(plan: dict) -> dict:
    from eb.runners.rlinf_ufo_runtime import (
        NATIVE_ROOT,
        OUTPUT_ROOT,
        configure_environment,
        file_hash,
        parameter_change,
        parameter_snapshot,
        parse_native_args,
        prepare_segment,
        validate_plan,
        verify_full_native_state_finite,
        verify_source_refs,
        verify_training_data,
        write_json,
    )
    from eb.ufo_native_baseline import metric_differences
    from eb.ufo_training_checkpoint import (
        canonical_hash,
        read_native_checkpoint,
    )

    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("source_run_id") != SOURCE_RUN
        or plan.get("base_image_digest") != BASE_IMAGE
        or plan.get("comparison_rules") != RULES
        or plan.get("hash") != canonical_hash(plan)
    ):
        raise ValueError("Unapproved native baseline plan")
    from eb.ufo_native_baseline import (
        VERIFICATION_SCOPE,
        build_baseline_checkpoint_manifest,
        execution_training_plan,
    )

    training = execution_training_plan(plan, plan["run_id"])
    validate_plan(training)
    if (
        training["world_size"] != 4
        or training["target_updates"] != 112
        or training["start_updates"] != 0
        or training["identity"]["seed"] != 42
        or training["identity"]["buffer_size"] != 262144
        or training.get("resume")
    ):
        raise ValueError("Baseline supports only the frozen fresh four-rank 112-update case")
    verify_source_refs(training)
    configure_environment(training)
    verify_training_data(training)
    reference = read_native_checkpoint(Path("/reference"), SOURCE_RUN, verify_hashes=True)
    if (
        file_hash(Path("/reference/outputs/native/checkpoint-manifest.json"))
        != plan["source_manifest_sha256"]
    ):
        raise ValueError("Reference checkpoint identity changed")
    for relative, expected in plan["reference_files"].items():
        if file_hash(Path("/reference") / relative) != expected:
            raise ValueError("Reference runtime identity/configuration changed")
    args = parse_native_args(training)
    prepare_segment(training, args)
    write_json(OUTPUT_ROOT / "outputs/baseline-frozen-plan.json", plan)
    write_json(OUTPUT_ROOT / "outputs/resolved-native-args.json", args)
    # configure_environment intentionally selected the frozen native sources.
    # Prefix only the image-owned observer; no user-controlled host script.
    os.environ["PYTHONPATH"] = "/opt/eb-baseline:" + os.environ["PYTHONPATH"]
    os.environ["EB_UFO_BASELINE_OBSERVER"] = "1"
    from ufo.train import launch

    # Real launch -> elastic_launch -> original run_train -> original Trainer.
    # All four children see four GPUs; native local rank controls CUDA/EGL.
    launch(argparse.Namespace(**{**args, "robot_config": Path(args["robot_config"]), "gpu_ids": "all"}))
    os.environ.pop("EB_UFO_BASELINE_OBSERVER", None)

    import torch
    from ufo.agents.common.buffers.trajectory import ReplayBuffer
    from ufo.training.config import TrainConfig

    checkpoint = NATIVE_ROOT / "checkpoint"
    status = json.loads((checkpoint / "train_status.json").read_text())
    counts = {key: status[key] for key in RULES["counts"]}
    if counts != RULES["counts"] or (checkpoint / ".saving").exists():
        raise ValueError("Native launch did not finish the approved 112 updates")
    cfg = TrainConfig.model_validate(json.loads((NATIVE_ROOT / "config.json").read_text()))
    agent = cfg.agent.object_class.load(checkpoint, device="cpu")
    verify_full_native_state_finite(agent, rank=0, **counts)
    after = parameter_snapshot(agent._model)
    optimizer_steps = optimizer_layout(agent.optimizer_dict)
    source_optimizers = torch.load(
        Path(reference["path"]) / "optimizers.pth", map_location="cpu", weights_only=True
    )
    if optimizer_layout(source_optimizers) != optimizer_steps:
        raise ValueError("Native/Worker optimizer schema, groups or internal counters differ")
    del source_optimizers
    del agent
    import gc

    gc.collect()
    ranks, comparisons = [], []
    for rank in range(4):
        initial = json.loads((OUTPUT_ROOT / f"outputs/runtime/baseline-initial-rank-{rank}.json").read_text())
        if not initial["profiler_disabled_before_hashing"] or initial["optimizer_steps_before_train"] != 0:
            raise ValueError("Baseline observer did not disable before training")
        before = (
            initial["parameters"],
            {name: torch.tensor(value, dtype=torch.float32) for name, value in initial["samples"].items()},
        )
        change = parameter_change(before, after)
        replay = ReplayBuffer.load(checkpoint / "buffers" / f"train_rank_{rank}", device="cpu")
        # Native len is the time-axis cursor (18); size includes 1024 envs.
        if replay.size() != 18432:
            raise ValueError("Native rank Replay transition count differs")
        del replay
        rank_result = {key: initial[key] for key in ("rank", "world_size", "pid", "gpu_uuid", "seed")}
        rank_result.update(
            **counts,
            status="succeeded",
            replay_loader_verified=True,
            source_refs=training["source_refs"],
            shared_checkpoint_parameter_change=change,
        )
        ranks.append(rank_result)
        source = next(row for row in reference["rank_results"] if row["rank"] == rank)
        if change["before_sha256"] != source["parameter_change"]["before_sha256"]:
            raise ValueError("Native initial hash differs from the actual Worker")
        comparisons.append(
            {
                "rank": rank,
                "configuration_leaf_values_equal": True,
                "initial_hash_exact": True,
                "final_hash_equal": change["after_sha256"] == source["parameter_change"]["after_sha256"],
                "native_shared_checkpoint_parameter_change": change,
                "worker_parameter_change": source["parameter_change"],
            }
        )
    # The diagnostic wrapper image is recorded distinctly from its e122 base.
    manifest = build_baseline_checkpoint_manifest(
        NATIVE_ROOT, training, plan["run_id"], rank_observations=ranks
    )
    if {row["path"] for row in manifest["files"]} != {row["path"] for row in reference["files"]}:
        raise ValueError("Native/Worker checkpoint member layout differs")
    for relative in (
        "checkpoint/model/config.json",
        "checkpoint/model/init_kwargs.json",
        "checkpoint/config.json",
        "checkpoint/init_kwargs.json",
    ):
        if json.loads((NATIVE_ROOT / relative).read_text()) != json.loads(
            (Path(reference["native_path"]) / relative).read_text()
        ):
            raise ValueError("Native/Worker saved model schema differs")
    sync = json.loads((checkpoint / "distributed_sync.json").read_text())
    if any(
        len(sync.get(key, [])) != 4 or not all(__import__("math").isfinite(x) for x in sync[key])
        for key in ("rank_param_buffer_sum", "rank_param_buffer_sqsum", "rank_param_buffer_numel")
    ):
        raise ValueError("Native synchronization report does not cover four finite ranks")
    write_json(NATIVE_ROOT / "checkpoint-manifest.json", manifest)
    metrics = metric_differences(
        Path(reference["native_path"]) / "train_log.txt", NATIVE_ROOT / "train_log.txt"
    )
    write_json(OUTPUT_ROOT / "outputs/metrics-differences.json", metrics)
    from eb.runners.ufo_training_analysis import build_training_evidence

    build_training_evidence(OUTPUT_ROOT, training)
    report = {
        "comparison_rules": RULES,
        "ranks": comparisons,
        "counts": counts,
        "optimizer_internal_steps": optimizer_steps,
        "native_synchronization": sync,
        "structural_checks_passed": True,
        "numerical_equivalence_claimed": False,
        "quality_qualification": "diagnostic",
        "base_image_digest": BASE_IMAGE,
        "observer_sha256": file_hash(Path(__file__).with_name("ufo_native_baseline_observer.py")),
        "verification_scope": VERIFICATION_SCOPE,
        "final_state_check_scope": (
            "Shared rank0 CPU checkpoint model/optimizers; all-rank model sync and Replay loaders"
        ),
        "launcher": "unmodified ufo.train.launch -> elastic_launch -> run_train",
        "native_process_group_timeout_seconds": 7200,
        "job_wall_timeout_seconds": 1800,
    }
    write_json(OUTPUT_ROOT / "outputs/baseline-report.json", report)
    selected = [NATIVE_ROOT / row["path"] for row in manifest["files"]]
    selected += [NATIVE_ROOT / "checkpoint-manifest.json"]
    for prefix in ("outputs/runtime", "outputs/analysis"):
        selected += [p for p in (OUTPUT_ROOT / prefix).rglob("*") if p.is_file()]
    selected += [
        OUTPUT_ROOT / f"outputs/{name}"
        for name in (
            "baseline-report.json",
            "metrics-differences.json",
            "baseline-frozen-plan.json",
            "resolved-native-args.json",
            "robot-config-receipt.json",
            "metrics-index.json",
        )
    ]
    files = [
        {
            "path": str(p.relative_to(OUTPUT_ROOT / "outputs")),
            "bytes": p.stat().st_size,
            "sha256": file_hash(p).removeprefix("sha256:"),
        }
        for p in sorted(set(selected))
    ]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "run_id": plan["run_id"],
        "operation": "ufo_native_baseline_v1",
        "plan_hash": plan["hash"],
        "source_run_id": SOURCE_RUN,
        "image_digest": plan["image_digest"],
        "base_image_digest": BASE_IMAGE,
        "counts": counts,
        "comparison_rules": RULES,
        "structural_checks_passed": True,
        "numerical_equivalence_claimed": False,
        "quality_qualification": "diagnostic",
        "verification_scope": VERIFICATION_SCOPE,
        "files": files,
    }
    write_json(OUTPUT_ROOT / "outputs/baseline-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    try:
        run(plan)
    except BaseException as error:
        from eb.runners.rlinf_ufo_runtime import OUTPUT_ROOT, write_json

        (OUTPUT_ROOT / "outputs/baseline-receipt.json").unlink(missing_ok=True)
        write_json(
            OUTPUT_ROOT / "outputs/baseline-failure.json",
            {
                "run_id": plan.get("run_id"),
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:4000],
            },
        )
        raise


if __name__ == "__main__":
    main()
