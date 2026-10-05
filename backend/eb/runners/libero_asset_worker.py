"""Container-only acquisition of the fixed LIBERO evaluation benchmark bundle."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

REPO = "RLinf/LIBERO-assets"
REVISION = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
LIBERO_PACKAGE = "rlinf-libero==0.1.3"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def verify_bundle(root: Path, expected: str | None = None) -> dict:
    manifest_path = root / "benchmark-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("LIBERO benchmark manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != "eb/libero-benchmark/v1" or manifest.get("repo") != REPO
            or manifest.get("revision") != REVISION or manifest.get("libero_package") != LIBERO_PACKAGE
            or not isinstance(manifest.get("files"), list)
            or len(manifest["files"]) < 100 or manifest.get("suites") != {name: 10 for name in SUITES}):
        raise ValueError("LIBERO benchmark manifest is invalid")
    files = manifest["files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("sha256") or (expected and digest != expected):
        raise ValueError("LIBERO benchmark manifest hash changed")
    seen, total = set(), 0
    for item in files:
        relative = item.get("path", "") if isinstance(item, dict) else ""
        if not isinstance(relative, str):
            raise ValueError("LIBERO benchmark member path is invalid")
        parts = Path(relative).parts
        if (relative in seen or not parts or Path(relative).is_absolute() or ".." in parts
                or "\\" in relative or parts[0] not in {"assets", "bddl_files", "init_files"}
                or type(item.get("bytes")) is not int or item["bytes"] < 0):
            raise ValueError("LIBERO benchmark member is missing or unsafe")
        path = root
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("LIBERO benchmark member contains symlink")
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("LIBERO benchmark member is missing or unsafe")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item.get("sha256"):
                raise ValueError("LIBERO benchmark member hash changed")
        seen.add(relative)
        total += item["bytes"]
    if total != manifest.get("total_bytes"):
        raise ValueError("LIBERO benchmark total bytes changed")
    return manifest


def copy_tree(source: Path, target: Path, files: list[dict], prefix: str) -> None:
    if not source.is_dir():
        raise ValueError(f"LIBERO {prefix} directory is missing")
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"LIBERO {prefix} contains unsupported member")
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": f"{prefix}/{relative.as_posix()}",
                      "bytes": destination.stat().st_size, "sha256": digest})


def acquire(destination: Path) -> dict:
    if destination.exists():
        manifest = verify_bundle(destination)
        return result_for(destination, manifest)
    from libero.libero import benchmark
    import libero.libero as core
    if importlib.metadata.version("rlinf-libero") != "0.1.3":
        raise ValueError("LIBERO benchmark package version changed")
    staging = destination.parent / ".libero-benchmark-staging"
    if staging.exists():
        raise ValueError("unfinished LIBERO acquisition exists")
    staging.mkdir(parents=True)
    try:
        package = Path(core.__file__).resolve().parent
        image_receipt = Path("/opt/eb-runtime-assets.json")
        if image_receipt.is_file():
            receipt = json.loads(image_receipt.read_text())
            if receipt.get("libero_assets_repo") != REPO or receipt.get("libero_assets_revision") != REVISION:
                raise ValueError("LIBERO image asset revision differs from the benchmark")
            source = package / "assets"
            reference = sorted(receipt["files"], key=lambda item: item["path"])
            needed = sum(item["bytes"] for item in reference)
        else:
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
            siblings = [item for item in info.siblings if item.rfilename != ".gitattributes"]
            if not siblings or any(item.size is None for item in siblings):
                raise ValueError("LIBERO asset source lacks file-size metadata")
            needed = sum(item.size for item in siblings)
            source = Path(snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                            max_workers=1, token=False))
            reference = None
        if needed > 5 * 1024**3 or shutil.disk_usage(destination.parent).free < needed * 2 + 20 * 1024**3:
            raise ValueError("LIBERO assets exceed 5 GiB limit or disk reserve")
        files: list[dict] = []
        copy_tree(source, staging / "assets", files, "assets")
        if reference is not None and sorted(
            ({**item, "path": item["path"].removeprefix("assets/")} for item in files),
            key=lambda item: item["path"],
        ) != reference:
            raise ValueError("LIBERO image asset files differ from the pinned receipt")
        copy_tree(package / "bddl_files", staging / "bddl_files", files, "bddl_files")
        copy_tree(package / "init_files", staging / "init_files", files, "init_files")
        if len(files) < 100 or not any(p["path"].endswith(".bddl") for p in files):
            raise ValueError("LIBERO benchmark asset bundle is incomplete")
        suites = {}
        for name in SUITES:
            tasks = benchmark.get_benchmark(name)().tasks
            if len(tasks) != 10:
                raise ValueError(f"LIBERO {name} has unexpected task count")
            for task in tasks:
                bddl = staging / "bddl_files" / task.problem_folder / task.bddl_file
                initial = staging / "init_files" / task.problem_folder / task.init_states_file
                if not bddl.is_file() or not initial.is_file():
                    raise ValueError(f"LIBERO {name} lacks task or initial state: {task.bddl_file}")
            suites[name] = len(tasks)
        files.sort(key=lambda item: item["path"])
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = {"schema_version": "eb/libero-benchmark/v1", "repo": REPO,
                    "libero_package": LIBERO_PACKAGE,
                    "revision": REVISION, "files": files, "sha256": digest,
                    "total_bytes": sum(item["bytes"] for item in files), "suites": suites}
        (staging / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(destination)
        verify_bundle(destination, digest)
        return result_for(destination, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def result_for(destination: Path, manifest: dict) -> dict:
    return {"format": "libero_benchmark", "total_bytes": manifest["total_bytes"],
            "source_refs": [{"kind": "huggingface", "uri": REPO, "revision": REVISION}],
            "benchmark": {"sha256": manifest["sha256"], "suites": manifest["suites"]},
            "validation": {"status": "validated", "scope": "libero_benchmark_assets",
                           "training_ready": False, "evaluation_ready": True,
                           "storage_ref": str(destination), "errors": []}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination, output = Path(args.dest), Path(args.output)
    try:
        result = acquire(destination)
    except Exception as exc:
        result = {"validation": {"status": "invalid", "scope": "libero_benchmark_assets",
                                  "training_ready": False, "evaluation_ready": False,
                                  "errors": [str(exc)[:500]]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    os.replace(temporary, output)
    if result["validation"]["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
