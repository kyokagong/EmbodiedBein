"""Build-time correction for mujoco-warp 3.8.1's undefined UNKNOWN frame axis."""

import hashlib
import importlib.metadata
import json

EXPECTED_VERSIONS = {"mujoco-warp": "3.8.1", "warp-lang": "1.18.0", "mujoco": "3.8.0"}
EXPECTED_SHA256 = "2ba411f182ba5fb7d88e85050c55ef4055fafefe2d26555f429ecd4e0d80c19e"
BEFORE = (
    "  else:  # UNKNOWN\n"
    "    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])"
)
AFTER = BEFORE.replace("    axis =", "    xmat = wp.identity(3, dtype=wp.float32)\n    axis =")


def main():
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"Unsupported MuJoCo Warp patch versions: {versions}")
    # Distribution metadata locates the file without importing mujoco_warp: its
    # package initializer eagerly imports sensor and would cache the old code.
    path = importlib.metadata.distribution("mujoco-warp").locate_file("mujoco_warp/_src/sensor.py")
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("MuJoCo Warp sensor source differs from the reviewed 3.8.1 payload")
    source = original.decode()
    if source.count(BEFORE) != 1:
        raise RuntimeError("MuJoCo Warp UNKNOWN axis patch requires exactly one matching branch")
    result = source.replace(BEFORE, AFTER).encode()
    path.write_bytes(result)
    print(json.dumps({
        "patch": "mujoco-warp-3.8.1-unknown-frame-axis-identity",
        "versions": versions,
        "original_sha256": EXPECTED_SHA256,
        "patched_sha256": hashlib.sha256(result).hexdigest(),
        "replacements": 1,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
