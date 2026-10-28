"""CPU check of the version-locked native factory; scratch payloads are metadata.

No CUDA device or GPU work. Kernel AST, launch dimensions/parameters and capacity
are compared to the original source, while native allocation branches execute.
"""

import ast
import hashlib
import importlib.util
import os
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from mujoco_warp._src import collision_convex, collision_driver, types  # noqa: E402


class Attributes(NS):
    def __getattr__(self, key):
        return None


class Count:
    def __init__(self, values, device):
        self.values, self.device, self.reads = values, device, 0
        self.events = []

    def numpy(self):
        self.reads += 1
        self.events.append("read")
        return self.values


def check_factory(module, values, *, cuda=True, pool=False, capture=False, invalid=False, enabled=True):
    cap = 1048576
    device = NS(is_cuda=cuda, is_capturing=capture, stream=object())
    count = Count(np.asarray(values), device)
    counts = [0] * 55
    for pair in ((types.GeomType.BOX, types.GeomType.MESH),
                 (types.GeomType.HFIELD, types.GeomType.MESH)):
        counts[module.upper_trid_index(10, pair[0].value, pair[1].value)] = 1
    model = Attributes(geom_pair_type_count=counts, nmaxpolygon=76, nmaxmeshdeg=18,
                       opt=Attributes(ccd_iterations=50, disableflags=0))
    data = Attributes(naccdmax=cap, naconmax=cap, ncollision=count,
                      contact=Attributes())
    ctx = Attributes()
    rows, launches, syncs = [], [], []

    def allocate(shape=None, dtype=float, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)
        value = NS(shape=tuple(shape), dtype=dtype)
        rows.append(value)
        return value

    pairs = [key for key, value in collision_driver.MJ_COLLISION_TABLE.items()
             if value == types.CollisionType.CONVEX]
    indices = [module.upper_trid_index(10, a.value, b.value) for a, b in pairs]
    assert len(set(indices)) == len(indices)

    def synchronize(stream):
        assert stream is device.stream and count.reads == 0
        syncs.append(stream)
        count.events.append("sync")

    with (patch.object(wp, "empty", side_effect=allocate), patch.object(wp, "zeros", side_effect=allocate),
          patch.object(wp, "is_mempool_enabled", return_value=pool),
          patch.object(wp, "get_stream", return_value=device.stream),
          patch.object(wp, "synchronize_stream", side_effect=synchronize),
          patch.object(wp, "launch", side_effect=lambda *a, **k: launches.append(k))):
        if invalid:
            try:
                module.convex_narrowphase(model, data, ctx, pairs)
            except ValueError:
                assert not rows and not launches
                return
            raise AssertionError("invalid candidate count was accepted")
        module.convex_narrowphase(model, data, ctx, pairs)
    dynamic = enabled and cuda and not pool and not capture
    expected = max(1, int(values[0])) if dynamic else cap
    assert len(rows) == 18 and all(row.shape[0] == expected for row in rows[1:]), (expected, rows[1].shape)
    assert count.reads == int(dynamic) and len(syncs) == int(dynamic)
    assert count.events == (["sync", "read"] if dynamic else [])
    assert all(call["dim"] == cap for call in launches)
    # Both kernels retain naconmax/naccdmax followed by the exact candidate array.
    for call in launches:
        index = next(i for i, value in enumerate(call["inputs"]) if value is count)
        assert call["inputs"][index - 2:index] == [cap, cap]
    assert data.naccdmax == cap and data.naconmax == cap
    if dynamic:
        # One atomic per matching candidate gives each type's indices [0,Ntype).
        # Heightfield prisms reuse that index; they do not create new rows.
        candidates = int(values[0])
        for per_type in (0, candidates // 2, candidates):
            assert per_type == 0 or per_type - 1 < expected


def main():
    original = Path(collision_convex.__file__).read_text()
    patch_path = Path("/workspace/deploy/patches/mujoco-warp/size_eager_convex_scratch.py")
    managed = runpy.run_path(str(patch_path)) if patch_path.exists() else None
    if managed and hashlib.sha256(original.encode()).hexdigest() == managed["PATCHED_SHA256"]:
        assert original.count(managed["HELPERS"]) == 1
        assert original.count(managed["LOCAL_SHAPE"]) == 17
        local = "  scratch_rows = _eb_eager_scratch_rows(d)\n\n"
        assert original.count(local + managed["ANCHOR"]) == 1
        original = original.replace(managed["HELPERS"], "", 1).replace(local, "", 1)
        original = original.replace(managed["LOCAL_SHAPE"], managed["SHAPE"])
        assert hashlib.sha256(original.encode()).hexdigest() == managed["EXPECTED_SHA256"]
    source = managed["patch_source"](original) if managed else original
    if managed:
        old_ast, new_ast = ast.parse(original), ast.parse(source)
        for name in ("ccd_hfield_kernel_builder", "ccd_kernel_builder"):
            before = next(n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
            after = next(n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(before) == ast.dump(after), "native kernel source changed"
        def launches(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "launch"]
        assert launches(old_ast) == launches(new_ast), "launch or kernel parameters changed"
        assert hashlib.sha256(source.encode()).hexdigest() == managed["PATCHED_SHA256"]
        for mutated in (original + "\n", original.replace("shape=(d.naccdmax,", "shape=(1,", 1)):
            try:
                managed["patch_source"](mutated)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unreviewed source was accepted")
    with tempfile.TemporaryDirectory(prefix="eb-convex-source-") as directory:
        path = Path(directory) / "collision_convex.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("eb_convex_scratch_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if managed:
            check_factory(module, np.array([13]), enabled=False)
        if hasattr(module, "eb_enable_dynamic_scratch"):
            module.eb_enable_dynamic_scratch()
        for value in (0, 1, 137, 1048575, 1048576):
            check_factory(module, np.array([value], dtype=np.int32))
        for values in (np.array([-1]), np.array([1048577]), np.array([], dtype=np.int32),
                       np.array([1, 2]), np.array([1.5]), np.array([True])):
            check_factory(module, values, invalid=True)
        for config in ({"cuda": False}, {"capture": True}, {"pool": True}):
            check_factory(module, np.array([13]), **config)
        stats = module.eb_dynamic_scratch_evidence()
        assert stats["calls"] == 5 and stats["min_candidates"] == 0
        assert stats["max_candidates"] == stats["max_rows"] == 1048576
    print("NATIVE_DYNAMIC_SCRATCH_SHAPES_INDEX_BOUNDS_KERNELS_LAUNCHES_AND_CAPTURE_GUARDS_VERIFIED",
          flush=True)


if __name__ == "__main__":
    main()
