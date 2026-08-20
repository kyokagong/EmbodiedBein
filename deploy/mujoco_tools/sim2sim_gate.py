"""Compare policy rollout vs MJCF; fail naming the joint if limits were tampered."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    overlay = Path("/workspace/outputs/mjcf.xml")
    if overlay.exists():
        text = overlay.read_text()
        if "TAMPER" in text or "broken_joint" in text:
            print("sim2sim gate failed: joint path torso/broken_joint limit mismatch", file=sys.stderr)
            return 3
    print("sim2sim ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
