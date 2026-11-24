"""Version-locked E2048 eager scratch storage patch; native kernels stay intact."""

import hashlib
import importlib.metadata
import json

EXPECTED_VERSIONS = {"mujoco-warp": "3.8.1", "warp-lang": "1.18.0", "mujoco": "3.8.0"}
EXPECTED_SHA256 = "1932c52be6867bed6ae3265ae683102f54099ae278ea93c3dcd25ff1f5e9c8bf"
PATCHED_SHA256 = "76a67d46e313634964e29ff83b332d900a25d2f96d4801a02136882b6a7901d9"
SHAPE = "shape=(d.naccdmax,"
LOCAL_SHAPE = "shape=(scratch_rows,"
ANCHOR = "  # ccd collider count\n"
HELPER_ANCHOR = "vec_maxconpair = wp.types.vector(length=MJ_MAXCONPAIR, dtype=float)"
HELPERS = '''# EmbodiedBein storage adaptation: explicitly enabled only by frozen E2048.
_eb_dynamic_scratch_enabled = False
_eb_scratch_stats = {
  "calls": 0, "min_candidates": None, "max_candidates": 0,
  "last_candidates": None, "last_rows": None, "max_rows": 0,
  "last_capacity": None, "max_capacity": 0,
}


def eb_enable_dynamic_scratch():
  global _eb_dynamic_scratch_enabled
  _eb_dynamic_scratch_enabled = True


def eb_dynamic_scratch_evidence():
  return {"enabled": _eb_dynamic_scratch_enabled, **_eb_scratch_stats}


def _eb_eager_scratch_rows(d):
  device = d.ncollision.device
  # Check capture first: a captured/default/CPU path must not sync or read C.
  if (not _eb_dynamic_scratch_enabled or not device.is_cuda
      or device.is_capturing or wp.is_mempool_enabled(device)):
    return d.naccdmax
  wp.synchronize_stream(wp.get_stream(device))
  values = d.ncollision.numpy()
  if values.shape != (1,) or values.dtype.kind not in ("i", "u"):
    raise ValueError("MuJoCo Warp candidate count must be one integer")
  candidates = int(values[0])
  capacity = min(d.naconmax, d.naccdmax)
  if candidates < 0 or candidates > capacity:
    raise ValueError(f"MuJoCo Warp candidate count {candidates} outside [0, {capacity}]")
  # Keep original launches even at C=0; their candidate guards return immediately.
  rows = max(1, candidates)
  stats = _eb_scratch_stats
  stats["calls"] += 1
  stats["min_candidates"] = (candidates if stats["min_candidates"] is None
                             else min(stats["min_candidates"], candidates))
  stats["max_candidates"] = max(stats["max_candidates"], candidates)
  stats["last_candidates"], stats["last_rows"] = candidates, rows
  stats["max_rows"] = max(stats["max_rows"], rows)
  stats["last_capacity"] = capacity
  stats["max_capacity"] = max(stats["max_capacity"], capacity)
  return rows


'''


def render_source(source: str) -> str:
    if source.count(SHAPE) != 17 or source.count(ANCHOR) != 1 or source.count(HELPER_ANCHOR) != 1:
        raise RuntimeError("MuJoCo Warp scratch patch requires exactly 17 original allocation shapes")
    result = source.replace(SHAPE, LOCAL_SHAPE)
    result = result.replace(ANCHOR, "  scratch_rows = _eb_eager_scratch_rows(d)\n\n" + ANCHOR)
    return result.replace(HELPER_ANCHOR, HELPERS + HELPER_ANCHOR)


def patch_source(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("MuJoCo Warp convex source differs from the reviewed 3.8.1 payload")
    result = render_source(source)
    if hashlib.sha256(result.encode()).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("MuJoCo Warp patched convex output differs from the reviewed payload")
    return result


def main():
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"Unsupported MuJoCo Warp scratch patch versions: {versions}")
    # Distribution metadata locates source without importing/caching old kernels.
    path = importlib.metadata.distribution("mujoco-warp").locate_file("mujoco_warp/_src/collision_convex.py")
    result = patch_source(path.read_text())
    path.write_text(result)
    print(json.dumps({
        "patch": "mujoco-warp-3.8.1-eager-candidate-scratch-v1", "versions": versions,
        "original_sha256": EXPECTED_SHA256, "patched_sha256": PATCHED_SHA256,
        "allocation_shape_replacements": 17,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
