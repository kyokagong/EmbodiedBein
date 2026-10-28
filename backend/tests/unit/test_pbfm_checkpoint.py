import json

import pytest
from test_pbfm_training import plan_fixture

from eb.pbfm_training_checkpoint import build_pbfm_checkpoint_manifest, validate_pbfm_checkpoint


def fixture(tmp_path):
    root = tmp_path / "native"

    def write(name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) if isinstance(data, dict) else data)

    model = {
        "name": "FBcprAuxModel",
        "archi": {
            "z_dim": 256,
            "actor": {
                "name": "direct_depth",
                "input_filter": {
                    "key": ["state", "last_action", "history_actor", "heading", "velocity_command"]
                },
            },
        },
    }
    agent = {"name": "FBcprAuxAgent", "model": model}
    cfg = {
        "agent": agent,
        "distributed_world_size": 1,
        "prior_plane_envs": 0,
        "checkpoint_buffer": True,
        "checkpoint_rank_buffers": True,
    }
    shapes = {
        "state": [52],
        "last_action": [23],
        "history_actor": [300],
        "privileged_state": [358],
        "heading": [2],
        "velocity_command": [3],
        "depth_image": [8, 18, 32],
        "velocity_target_yaw": [3],
        "terrain_priv": [121],
    }
    kwargs = {
        "action_dim": 23,
        "obs_space": {"spaces": {key: {"shape": shape} for key, shape in shapes.items()}},
    }
    for name, value in {
        "config.json": cfg,
        "effective_config.json": cfg,
        "effective-environment.json": {"robot": "rpo_23dof"},
        "config.yaml": "robot: rpo",
        "source-adaptation.json": {
            "schema_version": "eb/pbfm-source-adaptation/v1",
            "patch_sha256": "d" * 64,
        },
        "train_log.txt": "timestep,loss\n0,1\n",
        "checkpoint/config.json": agent,
        "checkpoint/init_kwargs.json": kwargs,
        "checkpoint/model/config.json": model,
        "checkpoint/model/init_kwargs.json": kwargs,
        "checkpoint/optimizers.pth": "state",
        "checkpoint/model/model.safetensors": "state",
        "checkpoint/train_status.json": {
            "local_time": 18432,
            "global_time": 18432,
            "optimizer_steps": 112,
            "world_size": 1,
        },
        "checkpoint/buffers/train/config.json": {
            "explicit_next": True,
            "sparse_next": True,
            "compact_depth_history": True,
            "n_dim": 2,
            "depth_history_offsets": [35, 30, 25, 20, 15, 10, 5, 0],
        },
        "checkpoint/buffers/train/buffer.hdf5": "replay",
    }.items():
        write(name, value)
    plan = plan_fixture()
    rank = {
        "rank": 0,
        "world_size": 1,
        "pid": 123,
        "gpu_uuid": "GPU-0",
        "status": "succeeded",
        "source_refs": plan["source_refs"],
        "local_time": 18432,
        "global_time": 18432,
        "optimizer_steps": 112,
        "loader_verified": True,
        "replay_loader_verified": True,
        "full_state_finite_verified": True,
        "strict_state_verified": True,
        "optimizer_names": [
            "actor_optimizer",
            "backward_optimizer",
            "forward_optimizer",
            "critic_optimizer",
            "discriminator_optimizer",
            "aux_critic_optimizer",
            "velocity_critic_optimizer",
            "heading_critic_optimizer",
        ],
    }
    return root, plan, rank


def test_complete_checkpoint_and_hash_mismatch(tmp_path):
    root, plan, rank = fixture(tmp_path)
    manifest = build_pbfm_checkpoint_manifest(
        root, plan, "run_test", rank_results=[rank], loader_verified=True
    )
    assert (
        validate_pbfm_checkpoint(root, manifest, run_id="run_test", plan=plan, verify_hashes=True)["step"]
        == 112
    )
    (root / "checkpoint/optimizers.pth").write_text("wrong")
    with pytest.raises(ValueError):
        validate_pbfm_checkpoint(root, manifest, run_id="run_test", plan=plan, verify_hashes=True)


@pytest.mark.parametrize(
    "failure",
    ["missing_shard", "half_write", "symlink", "missing_rank", "missing_optimizer", "wrong_identity"],
)
def test_incomplete_checkpoint_cannot_publish(tmp_path, failure):
    root, plan, rank = fixture(tmp_path)
    if failure == "missing_shard":
        (root / "checkpoint/buffers/train/buffer.hdf5").unlink()
    if failure == "half_write":
        (root / "checkpoint/.saving").touch()
    if failure == "symlink":
        (root / "link").symlink_to(root / "config.json")
    if failure == "missing_rank":
        rank["rank"] = 1
    if failure == "missing_optimizer":
        rank["optimizer_names"].pop()
    if failure == "wrong_identity":
        rank["source_refs"] = {}
    with pytest.raises(ValueError):
        build_pbfm_checkpoint_manifest(root, plan, "run_test", rank_results=[rank], loader_verified=True)
