from __future__ import annotations

import os


async def maybe_llm(text: str) -> str | None:
    """Only called when rules+playbooks miss. Unconfigured → never call."""
    base = os.environ.get("EB_LLM_BASE_URL")
    if not base:
        return None
    try:
        import httpx

        r = await httpx.AsyncClient(timeout=20.0).post(
            f"{base.rstrip('/')}/v1/chat/completions",
            json={"model": os.environ.get("EB_LLM_MODEL", "local"), "messages": [{"role": "user", "content": text}]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None
