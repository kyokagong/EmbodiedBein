"""Shared byte proxy; public routes retain their own visibility checks."""
import json

import anyio
import httpx
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from eb.settings import get_settings


def media_client():
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=get_settings().ebd_media_uds, verify=False), timeout=60.0)


async def stream_file(request, url: str, *, params: dict | None = None):
    client = media_client()
    try:
        response = await client.send(client.build_request(request.method, url, params=params,
            headers={key: request.headers[key] for key in ("range",) if key in request.headers}), stream=True)
    except BaseException as exc:
        with anyio.CancelScope(shield=True):
            await client.aclose()
        if isinstance(exc, httpx.HTTPError):
            raise HTTPException(502, "ebd 文件通道不可用") from exc
        raise

    async def close():
        with anyio.CancelScope(shield=True):
            try:
                await response.aclose()
            finally:
                await client.aclose()

    headers = {key: value for key, value in response.headers.items()
               if key in {"content-type", "content-length", "content-range", "accept-ranges", "cache-control"}}
    if request.method == "HEAD":
        return Response(status_code=response.status_code, headers=headers, background=BackgroundTask(close))

    async def chunks():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await close()

    return StreamingResponse(chunks(), status_code=response.status_code, headers=headers, background=BackgroundTask(close))


async def stream_storage_file(request, root: dict, path: str):
    return await stream_file(request, "http://ebd/v1/storage/file", params={"root": json.dumps(root), "path": path})
