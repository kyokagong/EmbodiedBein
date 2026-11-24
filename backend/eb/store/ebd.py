from __future__ import annotations

import re
from pathlib import Path
import httpx

from eb.settings import get_settings


def artifact_root(root: Path) -> dict:
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Artifact root must be an absolute managed Run path")
    settings = get_settings()
    for namespace, attribute in (("run", "data_root"), ("model_run", "models_root")):
        base = getattr(settings, attribute)
        try:
            parts = root.relative_to(Path(base) / "runs").parts
        except ValueError:
            continue
        if parts and re.fullmatch(r"run_[A-Za-z0-9_-]+", parts[0]):
            return {"namespace": namespace, "run_id": parts[0], "path": "/".join(parts[1:])}
    raise ValueError("Artifact root is outside managed Run storage")


def storage_root(root: Path) -> dict:
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Storage path must be absolute and canonical")
    settings = get_settings()
    for namespace, attribute in (("models", "models_root"), ("data", "datasets_root"),
                                  ("assets", "assets_root"), ("state", "data_root")):
        base = getattr(settings, attribute, None)
        if base is not None and root.is_relative_to(base):
            return {"namespace": namespace, "path": "" if root == Path(base) else root.relative_to(base).as_posix()}
    for prefix in sorted(getattr(settings, "allowed_path_prefixes", "").split(","), key=len, reverse=True):
        if prefix.strip() and root.is_relative_to(prefix.strip()):
            return {"namespace": "source", "source": prefix.strip(),
                    "path": "" if root == Path(prefix.strip()) else root.relative_to(prefix.strip()).as_posix()}
    raise ValueError("Storage path is outside configured roots")


async def storage_inventory(root: Path, *, recursive: bool = True, exclude_dirs: tuple[str, ...] = ()) -> list[dict]:
    reference = storage_root(root)
    result, cursors = [], set()
    cursor = None
    while True:
        page = await EbdClient().inspect_storage(reference, "", mode="inventory", cursor=cursor,
                                                limit=1000, recursive=recursive, exclude_dirs=exclude_dirs)
        result.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return result
        if cursor in cursors or len(result) > 100000:
            raise httpx.RemoteProtocolError("Unbounded ebd inventory pagination")
        cursors.add(cursor)


class EbdClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        # This is plain HTTP over a local Unix socket; no TLS context is needed.
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self.settings.ebd_uds, verify=False), timeout=timeout
        )

    async def verify_artifacts(self, root: dict, files: list[dict], *, max_files: int = 200,
                               max_file_bytes: int = 2 * 1024**3) -> dict:
        if (type(max_files) is not int or not 1 <= max_files <= 10000
                or type(max_file_bytes) is not int or not 1 <= max_file_bytes <= 1024**4
                or not isinstance(files, list) or not 1 <= len(files) <= max_files
                or any(not isinstance(item, dict) or type(item.get("bytes")) is not int
                       or not 0 <= item["bytes"] <= max_file_bytes for item in files)):
            raise ValueError("Invalid bounded artifact verification request")
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/artifacts/verify", json={
                "root": root, "files": files, "max_files": max_files, "max_file_bytes": max_file_bytes,
            })
            response.raise_for_status()
            try:
                result = response.json()
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/artifact-verification/v1"
                        or result.get("verified") is not True or type(result.get("file_count")) is not int
                        or result["file_count"] != len(files) or type(result.get("total_bytes")) is not int
                        or result["total_bytes"] != sum(item["bytes"] for item in files)):
                    raise ValueError("Artifact verification response differs from request")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd artifact verification response", request=response.request) from exc
            return result

    async def publish_file(self, root: dict, path: str, *, expected_bytes: int,
                           expected_sha256: str | None = None) -> dict:
        if (type(expected_bytes) is not int or not 0 <= expected_bytes <= 1024**4
                or expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
            raise ValueError("Invalid artifact publication expectation")
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/artifacts/publish-file", json={
                "root": root, "path": path, "expected_bytes": expected_bytes, "expected_sha256": expected_sha256,
            })
            response.raise_for_status()
            try:
                result = response.json()
                digest = result.get("sha256") if isinstance(result, dict) else None
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/artifact-publication/v1"
                        or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                        or expected_sha256 is not None and digest != expected_sha256
                        or type(result.get("size_bytes")) is not int or result["size_bytes"] != expected_bytes
                        or result.get("storage") != str(self.settings.data_root / "cas" / digest[:2] / digest)):
                    raise ValueError("Artifact publication response differs from request")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd artifact publication response", request=response.request) from exc
            return result

    async def inspect_storage(self, root: dict, path: str, *, mode: str, cursor: str | None = None,
                              limit: int = 50, recursive: bool = False, exclude_dirs: tuple[str, ...] = ()) -> dict:
        if mode not in {"inventory", "size", "disk"} or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid storage inspection request")
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/storage/inspect", json={
                "root": root, "path": path, "mode": mode, "cursor": cursor,
                "limit": limit, "recursive": recursive, "exclude_dirs": list(exclude_dirs),
            })
            response.raise_for_status()
            try:
                result = response.json()
                valid = isinstance(result, dict) and result.get("schema_version") == "eb/storage-inspection/v1" and result.get("mode") == mode
                if mode == "inventory":
                    valid = (valid and isinstance(result.get("items"), list) and len(result["items"]) <= limit
                             and type(result.get("total")) is int and result["total"] >= len(result["items"])
                             and (result.get("next_cursor") is None or isinstance(result["next_cursor"], str))
                             and all(isinstance(item, dict) and isinstance(item.get("path"), str)
                                     and item["path"] and len(item["path"].encode()) <= 1024
                                     and not any(part in {"", ".", ".."} for part in item["path"].split("/"))
                                     and "\\" not in item["path"] and not any(ord(c) < 32 for c in item["path"])
                                     and type(item.get("is_dir")) is bool and type(item.get("is_symlink")) is bool
                                     and type(item.get("bytes")) is int and item["bytes"] >= 0
                                     and type(item.get("mtime")) is int for item in result["items"]))
                else:
                    keys = ("bytes",) if mode == "size" else ("free_bytes", "total_bytes")
                    valid = valid and all(type(result.get(key)) is int and result[key] >= 0 for key in keys)
                if not valid:
                    raise ValueError("Invalid storage inspection response")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd storage inspection response", request=response.request) from exc
            return result

    async def read_storage_text(self, root: dict, path: str, *, max_bytes: int = 1024**2) -> str:
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1024**2:
            raise ValueError("Invalid bounded text limit")
        async with self._client() as client:
            response = await client.post("http://ebd/v1/storage/read-text", json={"root": root, "path": path, "max_bytes": max_bytes})
            response.raise_for_status()
            try:
                result = response.json()
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/storage-text/v1"
                        or not isinstance(result.get("text"), str) or len(result["text"].encode()) > max_bytes):
                    raise ValueError("Invalid storage text response")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd storage text response", request=response.request) from exc
            return result["text"]

    async def hash_storage_file(self, root: dict, path: str) -> dict:
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/storage/hash-file", json={"root": root, "path": path})
            response.raise_for_status()
            try:
                result = response.json()
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/storage-hash/v1"
                        or type(result.get("bytes")) is not int or not 0 <= result["bytes"] <= 1024**4
                        or not isinstance(result.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", result["sha256"])):
                    raise ValueError("Invalid storage hash response")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd storage hash response", request=response.request) from exc
            return result

    async def prepare_resume_files(self, source_run_id: str, target_run_id: str, files: list[dict]) -> dict:
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/storage/prepare-resume", json={
                "source_run_id": source_run_id, "target_run_id": target_run_id, "files": files})
            response.raise_for_status()
            try:
                result = response.json()
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/resume-preparation/v1"
                        or result.get("prepared") is not True or result.get("source_run_id") != source_run_id
                        or result.get("target_run_id") != target_run_id or not isinstance(result.get("files"), list)
                        or len(result["files"]) != len(files)):
                    raise ValueError("Invalid resume preparation response")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd resume preparation response", request=response.request) from exc
            return result

    async def pack_bundle(self, root: dict, members: list[dict]) -> dict:
        async with self._client(timeout=3600.0) as client:
            response = await client.post("http://ebd/v1/artifacts/pack-bundle", json={"root": root, "members": members})
            response.raise_for_status()
            try:
                result = response.json()
                digest = result.get("sha256") if isinstance(result, dict) else None
                if (not isinstance(result, dict) or result.get("schema_version") != "eb/artifact-publication/v1"
                        or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                        or type(result.get("size_bytes")) is not int or result["size_bytes"] <= 0
                        or result.get("storage") != str(self.settings.data_root / "cas" / digest[:2] / digest)):
                    raise ValueError("Invalid bundle publication response")
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Invalid ebd bundle publication response", request=response.request) from exc
            return result

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
