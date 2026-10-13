"""CPU transition replay and train-only contiguous UFO expert batches.

Storage deliberately differs from upstream trajectory replay; sampling semantics
and this implementation's complete state are frozen by the recipe.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

try:
    from eb.runners.bfm_ufo_agent import AUX_NAMES, OBS_DIMS, VELOCITY_AUX_NAMES
except ModuleNotFoundError:
    from bfm_ufo_agent import AUX_NAMES, OBS_DIMS, VELOCITY_AUX_NAMES


def tree_map(fn, value):
    return {k: tree_map(fn, v) for k, v in value.items()} if isinstance(value, dict) else fn(value)


def numpy_tree(value):
    return tree_map(lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x), value)


def tensor_tree(value):
    import torch

    return tree_map(lambda x: torch.from_numpy(x.copy()), value)


def flatten(tree, prefix=()):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from flatten(value, (*prefix, key))
        else:
            yield (*prefix, key), value


def unflatten(items):
    root = {}
    for keys, value in items:
        node = root
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return root


class UFOReplayBridge:
    def __init__(self, *, capacity=100_000, seed=0, aux_names=AUX_NAMES):
        if type(capacity) is not int or not 1 <= capacity <= 5_120_000:
            raise ValueError("CPU replay capacity must be 1–5120000")
        self.aux_names = tuple(aux_names)
        if self.aux_names not in (AUX_NAMES, VELOCITY_AUX_NAMES):
            raise ValueError("Unknown UFO replay auxiliary layout")
        self.capacity, self.count, self.cursor, self.received = capacity, 0, 0, 0
        self.storage = {}
        self.rng = np.random.default_rng(seed)

    def append(self, transitions: dict) -> int:
        arrays = dict(flatten(numpy_tree(transitions)))
        self._validate_arrays(arrays)
        size = len(arrays[("action",)])
        if self.storage and (
            self.storage.keys() != arrays.keys()
            or any(
                a.shape[1:] != self.storage[k].shape[1:] or a.dtype != self.storage[k].dtype
                for k, a in arrays.items()
            )
        ):
            raise ValueError("Replay schema changed")
        if not self.storage:
            self.storage = {
                k: np.empty((self.capacity, *a.shape[1:]), dtype=a.dtype) for k, a in arrays.items()
            }
        kept = min(size, self.capacity)
        indices = (self.cursor + np.arange(size - kept, size)) % self.capacity
        for key, array in arrays.items():
            self.storage[key][indices] = array[-kept:]
        self.cursor = (self.cursor + size) % self.capacity
        self.count = min(self.capacity, self.count + size)
        self.received += size
        return size

    def _validate_arrays(self, arrays):
        required = {
            ("action",),
            ("z",),
            ("episode_id",),
            ("step_id",),
            ("policy_version",),
            ("next", "terminated"),
            ("next", "truncated"),
        }
        required |= {("observation", key) for key in OBS_DIMS} | {
            ("next", "observation", key) for key in OBS_DIMS
        }
        required |= {("aux_rewards", key) for key in self.aux_names}
        if required != arrays.keys():
            raise ValueError("Replay lacks behavior/terminal/episode/policy fields")
        size = len(arrays[("action",)])
        if size < 1:
            raise ValueError("Empty replay append")
        for name, array in arrays.items():
            if (
                array.ndim < 1
                or len(array) != size
                or array.dtype.kind not in "fib"
                or (array.dtype.kind == "f" and not np.isfinite(array).all())
            ):
                raise ValueError(f"Malformed replay field {name}")
            if name in {("next", "terminated"), ("next", "truncated")} and (
                array.dtype != bool or array.shape != (size, 1)
            ):
                raise ValueError("Replay terminal flags must be [batch,1] bool")
        if arrays[("action",)].shape != (size, 23) or arrays[("z",)].shape != (size, 256):
            raise ValueError("Replay action/z shape differs from UFO recipe")
        for key, width in OBS_DIMS.items():
            if arrays[("observation", key)].shape != (size, width) or arrays[
                ("next", "observation", key)
            ].shape != (size, width):
                raise ValueError("Replay observation layout differs")
        for name in ("episode_id", "step_id", "policy_version"):
            if (
                arrays[(name,)].shape != (size, 1)
                or arrays[(name,)].dtype != np.int64
                or (arrays[(name,)] < 0).any()
            ):
                raise ValueError("Replay identity must be [batch,1] nonnegative int64")
        if any(
            a.dtype != np.float32
            for k, a in arrays.items()
            if k
            not in {
                ("episode_id",),
                ("step_id",),
                ("policy_version",),
                ("next", "terminated"),
                ("next", "truncated"),
            }
        ):
            raise ValueError("Replay numeric fields must be float32")
        if (np.abs(arrays[("action",)]) > 1).any() or any(
            arrays[("aux_rewards", key)].shape != (size, 1) for key in self.aux_names
        ):
            raise ValueError("Replay action bounds or raw aux layout differs")

    def sample_numpy(self, batch_size):
        if type(batch_size) is not int or batch_size < 1 or not self.count:
            raise ValueError("Replay is empty or batch_size invalid")
        indices = self.rng.integers(self.count, size=batch_size)
        return unflatten((k, a[indices].copy()) for k, a in self.storage.items())

    def sample(self, batch_size):
        return tensor_tree(self.sample_numpy(batch_size))

    def state_dict(self):
        return {**self.metadata(), "storage": unflatten((k, a[: self.count].copy()) for k, a in self.storage.items())}

    def metadata(self):
        return {
            "schema_version": "eb/ufo-cpu-replay/v2"
            if self.aux_names == VELOCITY_AUX_NAMES
            else "eb/ufo-cpu-replay/v1",
            **({"aux_names": list(self.aux_names)} if self.aux_names == VELOCITY_AUX_NAMES else {}),
            "capacity": self.capacity,
            "count": self.count,
            "cursor": self.cursor,
            "received": self.received,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def save_arrays(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = []
        for index, (key, value) in enumerate(sorted(self.storage.items())):
            filename = f"array_{index:03}.npy"
            data = value[:self.count]
            with (directory / filename).open("xb") as stream:
                np.lib.format.write_array_header_2_0(stream, {
                    "descr": np.lib.format.dtype_to_descr(data.dtype), "fortran_order": False,
                    "shape": data.shape})
                rows = max(1, 64 * 1024**2 // max(1, data[0:1].nbytes))
                for start in range(0, self.count, rows):
                    stream.write(memoryview(data[start:start+rows]).cast("B"))
            arrays.append({"key": list(key), "path": filename, "shape": list(data.shape),
                           "dtype": data.dtype.str})
        return {**self.metadata(), "arrays": arrays}

    def load_arrays(self, directory, state):
        directory = Path(directory)
        metadata = dict(state)
        members = metadata.pop("arrays", None)
        if (set(metadata) != set(self.metadata()) or metadata["capacity"] != self.capacity
                or metadata["schema_version"] != self.metadata()["schema_version"]
                or metadata.get("aux_names") != self.metadata().get("aux_names")
                or not isinstance(members, list) or len(members) > 100
                or any(type(metadata[k]) is not int for k in ("count", "cursor", "received"))
                or not 0 <= metadata["count"] <= self.capacity
                or metadata["count"] != min(self.capacity, metadata["received"])
                or metadata["cursor"] != metadata["received"] % self.capacity):
            raise ValueError("Replay arrays metadata/capacity differ")
        storage = {}
        for member in members:
            key = tuple(member["key"])
            if key in storage or member["path"] != f"array_{len(storage):03}.npy":
                raise ValueError("Duplicate or escaping replay member")
            path = directory / member["path"]
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing replay array")
            with path.open("rb") as stream:
                if np.lib.format.read_magic(stream) != (2, 0):
                    raise ValueError("Unknown replay array format")
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                if (list(shape) != member["shape"] or dtype.str != member["dtype"] or fortran
                        or not shape or shape[0] != metadata["count"] or dtype.kind not in "fib"
                        or path.stat().st_size != stream.tell() + int(np.prod(shape)) * dtype.itemsize):
                    raise ValueError("Partial or malformed replay array")
                target = np.empty((self.capacity, *shape[1:]), dtype=dtype)
                width = int(np.prod(shape[1:]))
                rows = max(1, 64 * 1024**2 // max(1, width * dtype.itemsize))
                for start in range(0, shape[0], rows):
                    count = min(rows, shape[0]-start)
                    target[start:start+count] = np.fromfile(stream,dtype=dtype,count=count*width).reshape(count,*shape[1:])
            storage[key] = target
        if metadata["count"]:
            # Validate in bounded blocks instead of making a full-pool finite mask.
            for start in range(0, metadata["count"], 4096):
                self._validate_arrays({key:value[start:min(start+4096,metadata["count"])] for key,value in storage.items()})
        elif storage:
            raise ValueError("Empty replay has unexpected arrays")
        self.storage = storage
        self.count, self.cursor, self.received = (metadata[k] for k in ("count", "cursor", "received"))
        self.rng.bit_generator.state = copy.deepcopy(metadata["rng"])

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("Unknown replay state fields")
        if (
            state["schema_version"] != expected["schema_version"]
            or state.get("aux_names") != expected.get("aux_names")
            or state["capacity"] != self.capacity
            or any(type(state[key]) is not int for key in ("capacity", "count", "cursor", "received"))
            or not 0 <= state["count"] <= self.capacity
            or not 0 <= state["cursor"] < self.capacity
            or state["received"] < state["count"]
            or state["cursor"] != state["received"] % self.capacity
            or state["count"] != min(self.capacity, state["received"])
        ):
            raise ValueError("Replay state counters/capacity differ")
        storage = dict(flatten(numpy_tree(state["storage"])))
        if state["count"]:
            self._validate_arrays(storage)
        elif storage:
            raise ValueError("Empty replay state has unexpected arrays")
        if any(len(value) != state["count"] for value in storage.values()):
            raise ValueError("Replay state count differs from arrays")
        self.storage = {k: np.empty((self.capacity, *a.shape[1:]), a.dtype) for k, a in storage.items()}
        for key, array in storage.items():
            self.storage[key][: state["count"]] = array
        self.count, self.cursor, self.received = state["count"], state["cursor"], state["received"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class UFOExpertSlicer:
    def __init__(self, clips, *, manifest_sha256, seed=0):
        ancestry = {}
        for clip in clips:
            group, split = clip["source_group"], clip["split"]
            if group in ancestry and ancestry[group] != split:
                raise ValueError("Ancestor group leaks across split")
            ancestry[group] = split
        self.clips = [clip for clip in clips if clip["split"] == "train"]
        self.manifest_sha256 = manifest_sha256
        self.device = "cpu"
        self.rng = np.random.default_rng(seed)
        self.source_weights = {"official_rpo": 0.5, "g1": 0.5}
        self.priorities = {}
        self.draws = {source: 0 for source in self.source_weights}
        self.windows = {}
        for length in (8, 250):
            self.windows[length] = {source: [] for source in self.source_weights}
            for index, clip in enumerate(self.clips):
                arrays, source = clip["arrays"], clip["source"]
                if source not in self.source_weights:
                    raise ValueError("Unknown expert source")
                valid = arrays["valid"]
                if valid.dtype != bool or len(valid) != len(arrays["states"]):
                    raise ValueError("Invalid expert valid mask")
                if len(valid) > length:
                    sums = np.convolve(valid.astype(int), np.ones(length + 1, int), "valid")
                    self.windows[length][source].extend(
                        (index, int(start)) for start in np.flatnonzero(sums == length + 1)
                    )
        if any(not self.windows[8][source] for source in self.source_weights):
            raise ValueError("Positive-weight source has no eligible train expert windows")

    def sample_numpy(self, batch_size, seq_length=8):
        if seq_length not in self.windows or batch_size < seq_length or batch_size % seq_length:
            raise ValueError("Expert batch must be divisible by approved sequence length")
        if any(not self.windows[seq_length][source] for source in self.source_weights):
            raise ValueError("Positive-weight source has no eligible long expert windows")
        values = {
            key: []
            for key in ("state", "privileged_state", "next_state", "next_privileged_state", "sample_ids")
        }
        sources = tuple(self.source_weights)
        draws = self.rng.choice(sources, size=batch_size // seq_length, p=list(self.source_weights.values()))
        for source in draws:
            windows = self.windows[seq_length][source]
            weights = np.asarray([self.priorities.get(self.clips[i]["id"], 1.0) for i, _ in windows])
            index, start = windows[int(self.rng.choice(len(windows), p=weights / weights.sum()))]
            arrays = self.clips[index]["arrays"]
            for native, local in (("state", "states"), ("privileged_state", "privileged_state")):
                values[native].append(arrays[local][start : start + seq_length])
                values["next_" + native].append(arrays[local][start + 1 : start + seq_length + 1])
            values["sample_ids"].append(
                np.stack((np.full(seq_length, index), np.arange(start, start + seq_length)), -1)
            )
            self.draws[source] += seq_length
        values = {k: np.concatenate(v).copy() for k, v in values.items()}
        return {
            "observation": {k: values[k] for k in ("state", "privileged_state")},
            "next": {"observation": {k: values["next_" + k] for k in ("state", "privileged_state")}},
            "sample_ids": values["sample_ids"],
        }

    def sample(self, batch_size, seq_length=8):
        return tree_map(
            lambda value: value.to(self.device), tensor_tree(self.sample_numpy(batch_size, seq_length))
        )

    def sample_at(self, clip_index, start_frame, seq_length=250):
        if (type(clip_index) is not int or type(start_frame) is not int
                or seq_length != 250 or not 0 <= clip_index < len(self.clips)
                or (clip_index, start_frame) not in self.windows[250][self.clips[clip_index]["source"]]):
            raise ValueError("Tracking requires a valid contiguous train reference window")
        arrays = self.clips[clip_index]["arrays"]
        self.draws[self.clips[clip_index]["source"]] += seq_length
        result = {"observation": {}, "next": {"observation": {}},
                  "sample_ids": np.stack((np.full(seq_length, clip_index),
                                          np.arange(start_frame, start_frame + seq_length)), -1)}
        for native, local in (("state", "states"), ("privileged_state", "privileged_state")):
            result["observation"][native] = arrays[local][start_frame:start_frame + seq_length]
            result["next"]["observation"][native] = arrays[local][start_frame + 1:start_frame + seq_length + 1]
        return tree_map(lambda value: value.to(self.device), tensor_tree(result))

    def set_priorities(self, priorities):
        allowed = {clip["id"] for clip in self.clips}
        if set(priorities) - allowed or any(not np.isfinite(v) or v <= 0 for v in priorities.values()):
            raise ValueError("Expert priorities require positive train-only tracking evidence")
        self.priorities = dict(priorities)

    def state_dict(self):
        return {
            "schema_version": "eb/ufo-expert-slicer/v1",
            "manifest_sha256": self.manifest_sha256,
            "source_weights": dict(self.source_weights),
            "priorities": dict(self.priorities),
            "draws": dict(self.draws),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if (
            set(state) != set(self.state_dict())
            or state["schema_version"] != "eb/ufo-expert-slicer/v1"
            or state["manifest_sha256"] != self.manifest_sha256
            or state["source_weights"] != self.source_weights
        ):
            raise ValueError("Expert source/schema/weights changed")
        self.set_priorities(state["priorities"])
        self.draws = dict(state["draws"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
