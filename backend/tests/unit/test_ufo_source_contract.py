import pytest

from eb.runners.bfm_ufo_agent import UFO_REVISION, validate_source_receipt


def receipt():
    return {"schema_version": "eb/ufo-image-source/v1",
        "ufo": {"commit": UFO_REVISION, "content_hash": "sha256:" + "a" * 64,
            "license": "CC-BY-NC-4.0"},
        "rlinf": {"commit": "b" * 40, "content_hash": "sha256:" + "c" * 64,
            "license": "Apache-2.0"}, "patch_hash": "sha256:" + "d" * 64,
        "packages": {"python": "3.11.16", "torch": "2.7.0+cu128", "pydantic": "2.11.5",
            "gymnasium": "1.2.0", "safetensors": "0.6.2", "tensordict": "0.11.0"}}


def test_agent_only_source_receipt():
    assert validate_source_receipt(receipt()) == receipt()
    for field in ("ufo", "rlinf", "packages", "patch_hash"):
        invalid = receipt()
        invalid.pop(field)
        with pytest.raises(ValueError):
            validate_source_receipt(invalid)
    invalid = receipt()
    invalid["ufo"]["commit"] = "e" * 40
    with pytest.raises(ValueError):
        validate_source_receipt(invalid)


def test_native_dependencies_are_not_workspace_dependencies():
    from pathlib import Path
    dockerfile = Path("../deploy/images/rlinf-ufo-agent.Dockerfile").read_text()
    assert "mjlab==" not in dockerfile
    assert "bfm_ufo_agent_smoke.py" in dockerfile
