from __future__ import annotations

import httpx

from eb.settings import get_settings


class EbdClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        # This is plain HTTP over a local Unix socket; no TLS context is needed.
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self.settings.ebd_uds, verify=False), timeout=timeout
        )

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

    async def image_info(self, name: str) -> dict:
        try:
            async with self._client() as c:
                r = await c.get("http://ebd/v1/images/inspect", params={"name": name})
                if r.status_code == 404:
                    return {"ok": False, "status": "missing"}
                r.raise_for_status()
                return r.json()
        except Exception as e:
            return {"ok": False, "status": "unavailable", "error": str(e)}

    async def submit_run(self, payload: dict) -> dict:
        async with self._client() as c:
            r = await c.post("http://ebd/v1/runs", json=payload)
            r.raise_for_status()
            return r.json()

    async def snapshot(self, path: str, content_hash: str | None = None) -> dict:
        async with self._client(timeout=3600.0) as c:
            r = await c.post("http://ebd/v1/snapshots", json={"path": path, "content_hash": content_hash})
            r.raise_for_status()
            return r.json()

    async def freeze_checkpoint(self, run_id: str, step: int) -> dict:
        async with self._client(timeout=3600.0) as c:
            r = await c.post("http://ebd/v1/checkpoints/freeze", json={"run_id": run_id, "step": step})
            r.raise_for_status()
            return r.json()

    async def get_run(self, run_id: str) -> dict:
        async with self._client() as c:
            r = await c.get(f"http://ebd/v1/runs/{run_id}")
            r.raise_for_status()
            return r.json()

    async def ps(self) -> dict:
        try:
            async with self._client() as c:
                r = await c.get("http://ebd/v1/ps")
                r.raise_for_status()
                return r.json()
        except Exception as e:
            return {"items": [], "error": str(e)}

    async def tsdb_query(self, run_id: str, tag: str | None = None) -> dict:
        async with self._client() as c:
            url = f"http://ebd/v1/runs/{run_id}/metrics"
            r = await c.get(url, params={"tag": tag} if tag else None)
            r.raise_for_status()
            return r.json()

    async def logs(self, run_id: str, tail: int = 200) -> dict:
        async with self._client() as c:
            r = await c.get(f"http://ebd/v1/runs/{run_id}/logs", params={"tail": tail})
            r.raise_for_status()
            return r.json()

    async def cancel_run(self, run_id: str) -> dict:
        async with self._client() as c:
            r = await c.post(f"http://ebd/v1/runs/{run_id}/cancel")
            r.raise_for_status()
            result = r.json()
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise RuntimeError("ebd cancellation was not confirmed")
            return result

    async def live_control(self, run_id: str, command: dict) -> dict:
        async with self._client(timeout=10.0) as c:
            r = await c.post(f"http://ebd/v1/runs/{run_id}/live/control", json=command)
            r.raise_for_status()
            return r.json()

    async def live_heartbeat(self, run_id: str) -> dict:
        async with self._client(timeout=10.0) as c:
            r = await c.post(f"http://ebd/v1/runs/{run_id}/live/heartbeat")
            r.raise_for_status()
            return r.json()

    async def delete_storage(self, paths: list[str], run_ids: list[str]) -> dict:
        async with self._client(timeout=3600.0) as c:
            r = await c.post("http://ebd/v1/storage/delete", json={"paths": paths, "run_ids": run_ids})
            r.raise_for_status()
            return r.json()

    async def scan(self, path: str) -> dict:
        async with self._client(timeout=3600.0) as c:
            r = await c.post("http://ebd/v1/scan", json={"path": path})
            r.raise_for_status()
            return r.json()
