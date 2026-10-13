"""Bounded native contact/codec qualification; no policy evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def phase(rank, output):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[rank + 1]
    os.environ["EB_UFO_DIAGNOSTICS_DIR"] = str(output / f"rank{rank}-startup")
    os.environ["WARP_CACHE_PATH"] = str(output / f"rank{rank}-warp")
    import numpy as np
    import torch
    from rpo_ufo_mjlab import RPOUFOMJLabPhysics
    env = RPOUFOMJLabPhysics(num_envs=2048, seed=42 + rank)
    try:
        initial = env.get_observations()
        repeated = env.get_observations()
        assert all(torch.equal(initial[k], repeated[k]) for k in initial), "observation read advanced state"
        anchor = env.last.clone()
        assert (initial["history_actor"] == 0).all()
        all_ids = torch.arange(2048, device="cuda:0")
        max_ratio = max_force = 0.
        first = total = first_resets = first_done = first_strict_done = 0
        alive = torch.ones(2048, dtype=torch.bool, device="cuda:0")
        traces, intervals, references = [], [], []
        for step in range(32):
            obs, _, terminated, truncated, _ = env.step(anchor)
            interval = obs["ufo_interval_trace"]
            assert interval.shape == (2048, 223) and (interval[:, 164] == 8).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert obs["states"].shape == (2048, 52) and obs["privileged_state"].shape == (2048, 358)
            max_ratio = max(max_ratio, float((interval[:, :23] / env.limits["joint_vel"]).max()))
            max_force = max(max_force, float(env.peak_force.max()))
            done = terminated | truncated
            reset_step = obs['step_id'].reshape(-1) == 1
            first_resets += int(reset_step.sum())
            first_done += int((done & reset_step).sum())
            first_strict_done += int((env.interval.strict_flags.any(-1) & reset_step).sum())
            if step == 0:
                first = int(done.sum())
            total += int(done.sum()); alive &= ~done
            traces.append(obs["ufo_action_trace"][:256].cpu().numpy())
            intervals.append(interval[:256].cpu().numpy())
            references.append(obs["ufo_reset_reference"][:256].cpu().numpy())
            if done.any():
                env.reset(env_ids=all_ids[done])
                anchor[done] = env.last[done]
                assert (env.get_observations()["history_actor"][done] == 0).all()
        file = output / f"rank{rank}.npz"
        np.savez_compressed(file, trace=np.stack(traces), interval=np.stack(intervals), reference=np.stack(references))
        result = {"rank": rank, "seed": 42 + rank, "num_envs": 2048, "steps": 32,
            "first_terminated": first, "total_terminated": total, "initial_episode_survivors": int(alive.sum()),
            "rsi_first_steps": first_resets, "rsi_first_terminated": first_done,
            "rsi_first_strict_violations": first_strict_done,
            "physics_profile": os.environ.get('EB_UFO_PHYSICS_PROFILE','native_mjcf_v12'),
            "max_physics_rated_ratio": max_ratio, "max_foot_force_N": max_force,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "file": file.name, "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        (output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


def validate_receipt(receipt, validation, output):
    if (receipt.get("schema_version") != "eb/ufo-mjlab-probe/v1"
            or receipt.get("qualification") != "physics_only"
            or receipt.get("dataset_manifest_sha256") != validation.get("manifest_sha256")
            or {r.get("rank") for r in receipt.get("ranks", [])} != {0, 1, 2}):
        raise ValueError("Native physics receipt identity differs")
    for row in receipt["ranks"]:
        path = output / f"rank{row['rank']}.npz"
        if (row.get("file") != path.name or row.get("num_envs") != 2048 or row.get("steps") != 32
                or path.is_symlink() or not path.is_file() or path.stat().st_size != row.get("bytes")
                or hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256")):
            raise ValueError("Native physics trace evidence differs")


def main():
    output = Path("/outputs")
    if len(sys.argv) == 2:
        phase(int(sys.argv[1]), output)
        return
    output.mkdir(parents=True, exist_ok=True)
    children = [subprocess.Popen([sys.executable, __file__, str(rank)]) for rank in range(3)]
    results = [child.wait() for child in children]
    if any(results):
        raise RuntimeError(f"Native physics workers failed: {results}")
    receipt = {"schema_version": "eb/ufo-mjlab-probe/v1", "qualification": "physics_only",
        "dataset_manifest_sha256": hashlib.sha256((Path(os.environ["EB_UFO_MOTION_ROOT"])/"manifest.json").read_bytes()).hexdigest(),
        "ranks": [json.loads((output/f"rank{rank}.json").read_text()) for rank in range(3)]}
    (output/"mjlab-probe.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
