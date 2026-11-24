"""CPU ledger from a frozen Run: real scene/model, allocation shapes only.

Mount the Run at /run/eb-output and RPO assets at /robot-assets. Run with
--runtime=runc, NVIDIA_VISIBLE_DEVICES=void, CUDA_VISIBLE_DEVICES=''. This never
constructs Simulation, Warp Data, the learner, or nworld-sized scratch arrays.
Logical allocation sizes do not measure CUDA graph physical residency/reuse.
"""

import dataclasses
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warp as wp  # noqa: E402
from mjlab.scene import Scene  # noqa: E402
from mujoco_warp._src import collision_convex, collision_driver, io, solver, types  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pbfm.envs.mjlab.config import make_mjlab_ufo_env_cfg  # noqa: E402


def main():
    root = Path("/run/eb-output")
    frozen = json.loads((root / "outputs/runtime/config-rank-0.json").read_text())
    hv = OmegaConf.create(json.loads(
        (root / "runtime/native-staging/effective-environment.json").read_text(),
    ))
    env = frozen["env"]
    cfg = make_mjlab_ufo_env_cfg(
        hv, num_envs=2048, seed=env["seed"], mjcf_path=env["mjcf_path"], auto_reset=False,
        robot_training=env["robot_training"],
    )
    scene = Scene(cfg.scene, device="cpu")
    mjm = scene.compile()
    cfg.sim.mujoco.apply(mjm)
    with wp.ScopedDevice("cpu"):
        model = io.put_model(mjm)
    assert all(device.is_cpu for device in wp.get_devices())
    assert (mjm.nq, mjm.nv, mjm.ngeom, model.nmaxpolygon, model.nmaxmeshdeg) == (30, 29, 817, 76, 18)
    assert not model.is_sparse and cfg.sim.nconmax == 512 and cfg.sim.njmax == 4096
    model.opt.ls_parallel = cfg.sim.ls_parallel
    world = 2048
    sizes = {"*": 1, **{f.name: getattr(mjm, f.name, None)
                        for f in dataclasses.fields(types.Model) if f.type is int}}
    sizes["nmaxcondim"] = int(max(mjm.geom_condim.max(), mjm.pair_dim.max(initial=0)))
    sizes["nmaxpyramid"] = max(1, 2 * (sizes["nmaxcondim"] - 1))
    sizes["njmax_pad"], sizes["nv_pad"] = io._get_padded_sizes(
        mjm.nv, cfg.sim.njmax, False, types.TILE_SIZE_JTDAJ_DENSE,
    )
    sizes.update(nworld=world, njmax=cfg.sim.njmax, naconmax=world * cfg.sim.nconmax)
    persistent = []
    for category, cls in (("data", types.Data), ("contact", types.Contact), ("efc", types.Constraint)):
        for field in dataclasses.fields(cls):
            shape = getattr(field.type, "shape", (0,))
            if shape == (0,):
                continue
            dims = tuple(sizes[d] if isinstance(d, str) else d for d in shape)
            persistent.append({"name": category + "." + field.name, "shape": dims,
                               "bytes": math.prod(dims) * wp.types.type_size_in_bytes(field.type.dtype)})
    for name, shape in (
        ("efc.J", (world, sizes["njmax_pad"], sizes["nv_pad"])),
        ("data.qM", (world, sizes["nv_pad"], sizes["nv_pad"])),
        ("data.qLD", (world, mjm.nv, mjm.nv)),
    ):
        persistent.append({"name": name, "shape": shape, "bytes": math.prod(shape) * 4})

    # Execute original scratch factory branches; substitute only allocator and
    # kernel boundaries. Objects below contain metadata, never array payloads.
    scratch, launch_inputs = [], []

    def allocate(shape=None, dtype=float, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)
        dtype = {bool: wp.bool, int: wp.int32, float: wp.float32}.get(dtype, dtype)
        frame = inspect.currentframe().f_back
        while frame and "mujoco_warp/_src/" not in frame.f_code.co_filename:
            frame = frame.f_back
        row = {"shape": tuple(shape), "bytes": math.prod(shape) * wp.types.type_size_in_bytes(dtype),
               "source": frame.f_code.co_filename, "line": frame.f_lineno}
        scratch.append(row)
        return row

    data = NS(nworld=world, njmax=cfg.sim.njmax, naccdmax=world * cfg.sim.nconmax,
              naconmax=world * cfg.sim.nconmax, nacon=None,
              ncollision=NS(device=wp.get_device("cpu")),
              geom_xpos=None, geom_xmat=None,
              contact=NS(**{name: None for name in types.Contact.__annotations__}))
    ctx = NS(collision_pair=None, collision_pairid=None, collision_worldid=None)
    pairs = [key for key, val in collision_driver.MJ_COLLISION_TABLE.items()
             if val == types.CollisionType.CONVEX]
    with (patch.object(wp, "empty", side_effect=allocate), patch.object(wp, "zeros", side_effect=allocate),
          patch.object(wp, "launch",
                       side_effect=lambda *args, **kwargs: launch_inputs.append(kwargs["inputs"]))):
        collision_convex.convex_narrowphase(model, data, ctx, pairs)
    convex = list(scratch)
    # One real non-heightfield launch receives EPA and multiccd simultaneously.
    largest_overlap = max(sum(row["bytes"] for row in convex if any(row is v for v in args))
                          for args in launch_inputs)
    assert largest_overlap >= (6976 + 7480) * 2**20
    scratch.clear()
    with patch.object(wp, "empty", side_effect=allocate), patch.object(wp, "zeros", side_effect=allocate):
        solver.create_solver_context(model, data)
    report = {
        "versions": {p: importlib.metadata.version(p)
                     for p in ("mjlab", "mujoco", "mujoco-warp", "warp-lang")},
        "model": {name: int(getattr(mjm, name))
                  for name in ("nq", "nv", "nu", "nbody", "ngeom", "nmesh", "nhfield")},
        "nmaxpolygon": int(model.nmaxpolygon), "nmaxmeshdeg": int(model.nmaxmeshdeg),
        "sizes": {k: sizes[k] for k in ("nworld", "naconmax", "njmax", "nv_pad", "njmax_pad")},
        "persistent_bytes": sum(r["bytes"] for r in persistent),
        "persistent_largest": sorted(persistent, key=lambda r: r["bytes"], reverse=True)[:20],
        "convex_scratch_bytes": sum(r["bytes"] for r in convex),
        "convex_scratch": convex, "same_kernel_overlap_bytes": largest_overlap,
        "solver_context_bytes": sum(r["bytes"] for r in scratch),
        "physical_graph_pool_bytes": "unmeasured; logical shapes do not establish CUDA pool residency",
    }
    print(json.dumps(report, indent=2, default=int), flush=True)
    print("FROZEN_CPU_MODEL_AND_NATIVE_SCRATCH_SHAPES_VERIFIED_NO_GPU_OR_LARGE_SCRATCH", flush=True)


if __name__ == "__main__":
    main()
