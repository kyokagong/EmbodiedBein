from __future__ import annotations

import httpx

from eb.settings import get_settings


class EbdClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=self.settings.ebd_uds), timeout=30.0)

    async def health(self) -> dict:
        try:
            async with self._client() as c:
                r = await c.get("http://ebd/healthz")
                r.raise_for_status()
                return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def doctor(self) -> dict:
        try:
            async with self._client() as c:
                r = await c.get("http://ebd/v1/doctor")
                r.raise_for_status()
                return r.json()
        except Exception as e:
            return {"gpus": [], "error": str(e)}

    async def submit_run(self, payload: dict) -> dict:
        async with self._client() as c:
            r = await c.post("http://ebd/v1/runs", json=payload)
            r.raise_for_status()
            return r.json()

    async def cancel_run(self, run_id: str) -> dict:
        async with self._client() as c:
            r = await c.post(f"http://ebd/v1/runs/{run_id}/cancel")
            r.raise_for_status()
            return r.json()

    async def scan(self, path: str) -> dict:
        async with self._client() as c:
            r = await c.post("http://ebd/v1/scan", json={"path": path})
            r.raise_for_status()
            return r.json()
