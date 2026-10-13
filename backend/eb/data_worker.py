"""Standalone data task; no API credentials, SQL or host container control."""
import argparse
import json
import shutil
from pathlib import Path

from eb.data_validation import (validate_amp_motion, validate_bfm_motion, validate_lerobot_v3,
                                validate_opendm_robotwin2, validate_ufo_robot_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--amp-asset-manifest")
    parser.add_argument("--bfm-asset-manifest")
    parser.add_argument("--ufo-asset-manifest")
    args = parser.parse_args()
    root, output = Path(args.src), Path(args.output)
    source = {"kind": "path", "uri": args.src}
    try:
        if args.repo_id and not args.local_only:
            import re
            from huggingface_hub import HfApi, snapshot_download
            if not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
                raise ValueError("HF revision must be a full commit SHA")
            files = HfApi().dataset_info(args.repo_id, revision=args.revision, files_metadata=True).siblings
            selected = [f for f in files if f.rfilename.startswith(("meta/", "data/", "videos/"))]
            if not selected or any(f.size is None for f in selected):
                raise ValueError("download requires known file sizes")
            needed = sum(f.size for f in selected)
            # ponytail: one 100 GiB task limit; shared filesystem leases needed for concurrent acquisition.
            if needed > 100 * 1024**3 or shutil.disk_usage(root).free < needed + 20 * 1024**3:
                raise ValueError("download exceeds 100 GiB limit or 20 GiB free-space reserve")
            snapshot_download(repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
                              local_dir=str(root), token=False,
                              allow_patterns=["meta/**", "data/**", "videos/**"])
        if args.repo_id:
            source = {"kind": "huggingface", "uri": args.repo_id, "revision": args.revision}
        result = (validate_ufo_robot_state(root, Path(args.ufo_asset_manifest), [])
                  if args.ufo_asset_manifest else
                  validate_amp_motion(root, Path(args.amp_asset_manifest), [source])
                  if args.amp_asset_manifest else
                  validate_bfm_motion(root, Path(args.bfm_asset_manifest), [source])
                  if args.bfm_asset_manifest else
                  validate_opendm_robotwin2(root, [source])
                  if (root / "index_cache.json").is_file() and (root / "norm_stats.json").is_file()
                  else validate_lerobot_v3(root, [source]))
    except Exception as exc:
        scope = ("rpo_ufo_robot_state_v1" if args.ufo_asset_manifest else
                 "rpo_bfm_motion_v1" if args.bfm_asset_manifest else
                 "rpo_amp_motion_v1" if args.amp_asset_manifest else "lerobot_v3_structure")
        result = {"validation": {"status": "invalid", "scope": scope,
                                  "training_ready": False, "errors": [str(exc)]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(output)
    if result["validation"]["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
