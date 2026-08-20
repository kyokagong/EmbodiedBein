from __future__ import annotations

from eb.domain.job import JobSpec


def fingerprint(spec: JobSpec, image_digest: str | None, commit: str | None) -> dict:
    datasets = spec.inputs.datasets if spec.inputs else []
    data_version = None
    if datasets:
        data_version = datasets[0].get("version") if isinstance(datasets[0], dict) else None
    return {
        "code_ref": commit,
        "config_hash": spec.canonical_hash(),
        "data_version": data_version,
        "image_digest": image_digest,
        "seed": spec.params.get("seed"),
    }
