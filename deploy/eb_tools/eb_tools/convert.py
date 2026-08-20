from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    out = Path("/workspace/outputs")
    out.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in Path(args.src).rglob("*") if _.is_file()) if Path(args.src).exists() else 0
    (out / "convert.json").write_text(f'{{"files": {n}, "run_id": "{args.run_id}"}}\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
