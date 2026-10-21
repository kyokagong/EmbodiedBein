import json
import sys
from types import SimpleNamespace

import pytest

from eb.runners import rlinf_ufo_runtime as runtime


def plan():
    return {
        "schema_version": "eb/ufo-native-train-plan/v1",
        "run_id": "run_test",
        "world_size": 2,
        "target_updates": 112,
        "fingerprint": "sha256:" + "a" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "native_args": [
            "--robot-config",
            "/sources/ufo/config.yaml",
            "--data-path",
            "/training-data/train.pkl",
            "--work-dir",
            "/run/eb-output/outputs/native",
            "--num-env-steps",
            "131072",
        ],
    }


@pytest.mark.parametrize(
    "extra",
    [
        ["--smoke"],
        ["--use-wandb"],
        ["--agent", "tech"],
        ["--gpu-ids", "all"],
        ["--num-env-steps", "2"],
        ["--data-manifest", "/tmp/data"],
    ],
)
def test_plan_rejects_unfrozen_or_duplicate_native_options(extra):
    value = plan()
    value["native_args"] += extra
    with pytest.raises(ValueError):
        runtime.validate_plan(value)


def test_plan_hash_binds_run_and_does_not_include_hash():
    value = plan()
    value["hash"] = runtime.canonical_plan_hash(value)
    assert runtime.validate_plan(value) is value
    value["run_id"] = "run_changed"
    with pytest.raises(ValueError, match="hash"):
        runtime.validate_plan(value)


@pytest.mark.parametrize("target", (16, 100, 113, 1_000_016))
def test_plan_bounds_actual_update_budget(target):
    value = plan()
    value["target_updates"] = target
    with pytest.raises(ValueError, match="target_updates"):
        runtime.validate_plan(value)


def test_plan_accepts_explicit_larger_update_capability_without_changing_segment_defaults():
    value = plan()
    value["target_updates"] = 1_000_000
    assert runtime.validate_plan(value) is value


def test_scoped_argv_restores_identity_even_on_native_parser_failure():
    previous = sys.argv
    with pytest.raises(RuntimeError):
        with runtime.scoped_argv(["--seed", "42"]):
            assert sys.argv == ["ufo.train", "--seed", "42"]
            raise RuntimeError("native failure")
    assert sys.argv is previous


def test_builder_forwards_native_run_train_arguments_without_new_defaults(monkeypatch):
    names = (
        "num_envs num_env_steps wandb_run_name checkpoint_every_steps buffer_every_steps "
        "data_path data_mix_weights update_z_every_step buffer_size lr_scale clip_grad_norm "
        "num_agent_updates robot_config network_width shared_critic agent "
        "disable_eval_prioritization smoke disable_dr disable_obs_noise cartwheel_aux_safe"
    ).split()
    args = {name: index for index, name in enumerate(names)}
    args.update(seed=4728, work_dir="/run/eb-output/outputs/native")
    monkeypatch.setitem(sys.modules, "ufo.train", SimpleNamespace(build_mjlab_config=lambda **kw: kw))
    result = runtime.build_native_config(args, rank=3, world_size=4)
    assert set(result) == set(names) | {
        "device",
        "work_dir",
        "seed",
        "use_wandb",
        "distributed_rank",
        "distributed_world_size",
    }
    assert result["seed"] == 4731
    assert result["distributed_rank"] == 3
    assert result["distributed_world_size"] == 4
    assert result["device"] == "cuda:0"
    for name in names[:16]:
        assert result[name] == args[name]


def test_save_stop_rejects_earlier_boundary_and_requires_aligned_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "NATIVE_ROOT", tmp_path / "native")
    args = {
        "num_env_steps": 8192,
        "num_envs": 1024,
        "checkpoint_every_steps": 4096,
        "buffer_every_steps": 8192,
    }
    with pytest.raises(ValueError, match="earlier"):
        runtime.prepare_segment(plan(), args)
    args["checkpoint_every_steps"] = 8192
    args["num_env_steps"] = 8193
    with pytest.raises(ValueError, match="align"):
        runtime.prepare_segment(plan(), args)
    args["num_env_steps"] = 8192
    runtime.prepare_segment(plan(), args)
    assert (tmp_path / "native/SAVE_AND_STOP").is_file()


def test_resume_copies_and_rejects_partial_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "NATIVE_ROOT", tmp_path / "native")
    source = tmp_path / "previous/checkpoint"
    source.mkdir(parents=True)
    (source / ".saving").touch()
    value = plan()
    value["resume"] = {"checkpoint_path": str(source)}
    args = {
        "num_env_steps": 16384,
        "num_envs": 1024,
        "checkpoint_every_steps": 8192,
        "buffer_every_steps": 8192,
    }
    with pytest.raises(ValueError, match="Incomplete"):
        runtime.prepare_segment(value, args)
    (source / ".saving").unlink()
    (source.parent / "checkpoint-manifest.json").write_text(json.dumps({"run_id": "run_previous"}))
    verified = []
    monkeypatch.setitem(
        sys.modules,
        "eb.ufo_training_checkpoint",
        SimpleNamespace(validate_native_checkpoint=lambda *args, **kwargs: verified.append(kwargs)),
    )
    status = {"world_size": 2, "local_time": 4096, "optimizer_steps": 112,
              "global_time": 8192, "buffer_global_time": 8192, "replay_id": "frozen"}
    (source / "train_status.json").write_text(json.dumps(status))
    for rank in range(2):
        shard = source / "buffers" / f"train_rank_{rank}"
        shard.mkdir(parents=True)
        (shard / "snapshot.json").write_text(
            json.dumps({"id": "frozen", "global_time": 8192, "world_size": 2, "transition_format": 2})
        )
    runtime.prepare_segment(value, args)
    copied = tmp_path / "native/checkpoint/train_status.json"
    assert copied.stat().st_ino != (source / "train_status.json").stat().st_ino
    assert copied.read_bytes() == (source / "train_status.json").read_bytes()
    assert verified == [{"run_id": "run_previous", "plan": value, "verify_hashes": True}]


def test_ray_rejects_inherited_external_cluster_before_import(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray://another-job:10001")
    with pytest.raises(ValueError, match="RAY_ADDRESS"):
        runtime.configure_environment(plan())


@pytest.mark.parametrize("probe, rank", ((False, 0), (True, 2), (True, -1)))
def test_failure_injection_never_becomes_a_training_plan_feature(probe, rank):
    from eb.runners.rlinf_ufo_entry import run

    with pytest.raises(ValueError, match="diagnostic probe rank"):
        run(plan(), probe=probe, probe_fail_rank=rank)


@pytest.mark.parametrize("failure_at", ("rank", "analysis"))
def test_failed_group_or_analysis_kills_all_owned_ranks_without_publishing(tmp_path, monkeypatch, failure_at):
    from eb.runners import rlinf_ufo_entry as entry

    for name in (
        "validate_plan",
        "verify_source_refs",
        "verify_training_data",
        "configure_environment",
        "prepare_segment",
    ):
        monkeypatch.setattr(entry, name, lambda *args: None)
    monkeypatch.setattr(entry, "parse_native_args", lambda value: {})
    monkeypatch.setattr(entry, "OUTPUT_ROOT", tmp_path)
    killed, shutdown = [], []
    ray = SimpleNamespace(
        _private=SimpleNamespace(
            worker=SimpleNamespace(
                _global_node=SimpleNamespace(
                    get_session_dir_path=lambda: str(tmp_path / "runtime/ray/session_test")
                )
            )
        ),
        nodes=lambda: [{"Alive": True, "NodeID": "local", "NodeManagerAddress": "127.0.0.1"}],
        kill=lambda actor, **kwargs: killed.append(actor),
        shutdown=lambda: shutdown.append(True),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    identities = [{"rank": rank, "gpu_uuid": str(rank)} for rank in range(2)]

    def fail():
        raise RuntimeError("rank failed")

    def complete_checkpoint(*args, **kwargs):
        runtime.write_json(tmp_path / "outputs/receipt.json", {"status": "complete"})
        return {"status": "complete"}

    def fail_analysis(*args):
        raise RuntimeError("analysis failed")

    monkeypatch.setitem(
        sys.modules,
        "eb.ufo_training_checkpoint",
        SimpleNamespace(write_native_checkpoint_receipt=complete_checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "eb.runners.ufo_training_analysis",
        SimpleNamespace(build_training_evidence=fail_analysis),
    )

    group = SimpleNamespace(
        worker_info_list=[SimpleNamespace(worker=rank) for rank in range(2)],
        describe_runtime=lambda: SimpleNamespace(wait=lambda: identities),
        run_segment=lambda *args, **kwargs: SimpleNamespace(
            wait=fail if failure_at == "rank" else lambda: [{"loader_verified": True}]
        ),
    )
    group.launch = lambda **kwargs: group

    def cluster(**kwargs):
        return SimpleNamespace(_distributed_log_collector=None)

    cluster.NAMESPACE = "test"
    monkeypatch.setitem(
        sys.modules,
        "rlinf.scheduler",
        SimpleNamespace(Cluster=cluster, PackedPlacementStrategy=lambda *args: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "eb.runners.rlinf_ufo_worker",
        SimpleNamespace(UfoTrainWorker=SimpleNamespace(create_group=lambda value: group)),
    )
    with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
        entry.run(plan())
    assert killed == [0, 1]
    assert shutdown == [True]
    assert not (tmp_path / "outputs/receipt.json").exists()


def test_training_hash_check_precedes_pickle_loading_and_forbids_holdout(tmp_path):
    value = plan()
    robot = tmp_path / "robot.yaml"
    xml = tmp_path / "robot.xml"
    xml.write_text("frozen XML")
    robot.write_text(f"xml_path: {xml}\n")
    value["native_args"][1] = str(robot)
    (tmp_path / "train.pkl").write_bytes(b"opaque native pickle")
    manifest = {
        "schema_version": "eb/ufo-native-motion/v1",
        "split": "train",
        "train_file": "train.pkl",
        "dataset_id": "ds_train",
        "native_loader": {"all_motions_loaded": True},
        "file_sha256": runtime.file_hash(tmp_path / "train.pkl").removeprefix("sha256:"),
        "robot_config_sha256": runtime.file_hash(robot).removeprefix("sha256:"),
        "robot_xml_sha256": runtime.file_hash(xml).removeprefix("sha256:"),
    }
    data = tmp_path / "data"
    data.mkdir()
    robot_pickle = tmp_path / "train.pkl"
    robot_pickle.rename(data / "train.pkl")
    (data / "manifest.json").write_text(json.dumps(manifest))
    value["identity"] = {
        "dataset_id": "ds_train",
        "dataset_manifest": runtime.file_hash(data / "manifest.json").removeprefix("sha256:"),
    }
    assert runtime.verify_training_data(value, data) == manifest
    (data / "train.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="pickle digest"):
        runtime.verify_training_data(value, data)
    (data / "validation.pkl").write_bytes(b"holdout")
    with pytest.raises(ValueError, match="only"):
        runtime.verify_training_data(value, data)


def test_weight_evidence_is_read_only_and_requires_a_finite_update():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    for name in ("_actor", "_forward_map", "_backward_map"):
        setattr(model, name, torch.nn.Linear(2, 2))
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    before = runtime.parameter_snapshot(model)
    assert all(torch.equal(value, original[name]) for name, value in model.named_parameters())
    with pytest.raises(ValueError, match="did not change"):
        runtime.parameter_change(before, runtime.parameter_snapshot(model))
    with torch.no_grad():
        model._actor.weight.add_(0.01)
    change = runtime.parameter_change(before, runtime.parameter_snapshot(model))
    assert change["before_sha256"] != change["after_sha256"]
    assert change["sample_delta_l2"] > 0
    assert change["sample_delta_max_abs"] == pytest.approx(0.01, abs=1e-6)
    with torch.no_grad():
        model._actor.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.parameter_snapshot(model)
