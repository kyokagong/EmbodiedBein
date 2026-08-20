from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from eb.settings import get_settings


def tape_path() -> Path:
    p = get_settings().data_root / "agent" / "tape.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_event(*, role: str, tool: str | None, args: dict, result: object, llm_calls: int = 0) -> None:
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "role": role,
        "tool": tool,
        "args": args,
        "result": result if isinstance(result, (str, int, float, dict, list, type(None))) else str(result),
        "llm_calls": llm_calls,
    }
    try:
        with tape_path().open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
