import hashlib
import json

import pytest

from eb.runners.libero_asset_worker import LIBERO_PACKAGE, REPO, REVISION, SUITES, acquire, verify_bundle


def test_pinned_libero_bundle_is_verified_before_reuse(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    files = []
    for index in range(100):
        path = root / "assets" / f"asset_{index}.xml"
        path.parent.mkdir(exist_ok=True)
        data = f"asset {index}".encode()
        path.write_bytes(data)
        files.append({"path": f"assets/{path.name}", "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"schema_version": "eb/libero-benchmark/v1", "repo": REPO, "revision": REVISION,
                "libero_package": LIBERO_PACKAGE,
                "files": files, "sha256": digest, "total_bytes": sum(f["bytes"] for f in files),
                "suites": {name: 10 for name in SUITES}}
    (root / "benchmark-manifest.json").write_text(json.dumps(manifest))
    assert acquire(root)["benchmark"]["sha256"] == digest
    (root / "assets/asset_1.xml").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_bundle(root, digest)
