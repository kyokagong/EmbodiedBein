"""Opt-in real ebd tests. Set EB_TEST_EBD to a container-built binary; no Docker socket needed."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess

import httpx
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def artifact_daemon(tmp_path, monkeypatch):
    executable = os.environ.get("EB_TEST_EBD")
    if not executable:
        pytest.skip("EB_TEST_EBD must name the isolated container-built ebd")
    from eb.settings import get_settings
    for name, value in {"EB_DATA_ROOT":str(tmp_path / "state"), "EB_MODELS_ROOT":str(tmp_path / "models"),
            "EB_DATASETS_ROOT":str(tmp_path / "data"), "EB_ASSETS_ROOT":str(tmp_path / "assets"),
            "EB_ALLOWED_PATH_PREFIXES":str(tmp_path / "source"),
            "EB_EBD_UDS":str(tmp_path / "control.sock"), "EB_EBD_MEDIA_UDS":str(tmp_path / "media.sock")}.items():
        monkeypatch.setenv(name, value)
    for directory in ("state/cas", "models", "data", "assets", "source"):
        (tmp_path / directory).mkdir(parents=True)
    get_settings.cache_clear()
    with (tmp_path / "ebd.log").open("w") as log:
        process = subprocess.Popen([executable, "serve"], env=os.environ.copy(), stdout=log, stderr=log)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError((tmp_path / "ebd.log").read_text())
                if (tmp_path / "control.sock").exists() and (tmp_path / "media.sock").exists():
                    break
                await asyncio.sleep(.05)
            else:
                raise RuntimeError("isolated ebd did not bind its sockets")
            yield tmp_path, process
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            get_settings.cache_clear()


@pytest.mark.asyncio
async def test_real_uds_hash_publication_inventory_and_media(artifact_daemon):
    from eb.store.ebd import EbdClient
    root, _ = artifact_daemon
    directory = root / "state/runs/run_integration/outputs"
    directory.mkdir(parents=True)
    (directory / "robot.urdf").write_bytes(b"<robot/>")
    (directory / "empty").write_bytes(b"")
    reference = {"namespace":"run","run_id":"run_integration","path":"outputs"}
    files = [{"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
             for name,data in (("robot.urdf",b"<robot/>"),("empty",b""))]
    client = EbdClient()
    assert (await client.verify_artifacts(reference, files))["file_count"] == 2
    identity = await client.publish_file(reference, "robot.urdf", expected_bytes=8, expected_sha256=files[0]["sha256"])
    assert Path(identity["storage"]).read_bytes() == b"<robot/>"
    assert await client.publish_file(reference, "robot.urdf", expected_bytes=8) == identity
    assert (await client.inspect_storage(reference, "", mode="size"))["bytes"] == 8
    assert (await client.hash_storage_file(reference,"robot.urdf"))["sha256"] == files[0]["sha256"]
    assert await client.read_storage_text(reference, "robot.urdf") == "<robot/>"
    assert (await client.inspect_storage(reference, "", mode="inventory", limit=1))["total"] == 2
    (directory / "linked").symlink_to(root / "state/cas")
    with pytest.raises(httpx.HTTPStatusError) as denied:
        await client.read_storage_text(reference,"linked/secret")
    assert denied.value.response.status_code == 403
    with pytest.raises(httpx.HTTPStatusError) as bad:
        await client.verify_artifacts(reference, [{**files[0],"sha256":"0"*64}])
    assert bad.value.response.status_code == 409
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(root / "media.sock")), base_url="http://ebd") as media:
        params = {"root":json.dumps(reference),"path":"robot.urdf"}
        response = await media.get("/v1/storage/file", params=params, headers={"range":"bytes=1-3"})
        assert response.status_code == 206 and response.content == b"rob"
        assert response.headers["content-type"] == "application/xml"
        response = await media.head("/v1/storage/file", params=params)
        assert response.content == b"" and response.headers["content-length"] == "8"
        response = await media.get("/v1/storage/file", params=params, headers={"range":"bytes=8-"})
        assert response.status_code == 416 and response.headers["content-range"] == "bytes */8"
        response = await media.get("/media/" + identity["sha256"][:2] + "/" + identity["sha256"])
        assert response.content == b"<robot/>"
        response = await media.get("/v1/storage/file", params={**params,"path":"empty"})
        assert response.content == b"" and response.headers["content-length"] == "0"
    Path(identity["storage"]).chmod(0o644)
    Path(identity["storage"]).write_bytes(b"xxxxxxxx")
    with pytest.raises(httpx.HTTPStatusError) as corrupt:
        await client.publish_file(reference,"robot.urdf",expected_bytes=8)
    assert corrupt.value.response.status_code == 409


@pytest.mark.asyncio
async def test_real_uds_pack_bundle_never_falls_back_to_raw_tar(artifact_daemon):
    from eb.store.ebd import EbdClient
    root,_ = artifact_daemon
    staging = root / "state/runs/run_bundle/outputs/bundle-staging"
    staging.mkdir(parents=True)
    (staging / "policy.onnx").write_bytes(b"checked model fixture")
    files = [{"path":"policy.onnx","bytes":21,"sha256":hashlib.sha256(b"checked model fixture").hexdigest()}]
    reference = {"namespace":"run","run_id":"run_bundle","path":"outputs/bundle-staging"}
    publication = await EbdClient().pack_bundle(reference,files)
    assert Path(publication["storage"]).read_bytes()[:4] == bytes.fromhex("28b52ffd")
    with pytest.raises(httpx.HTTPStatusError):
        await EbdClient().pack_bundle(reference,[{**files[0],"path":"../policy.onnx"}])


@pytest.mark.asyncio
async def test_artifact_io_measurements(artifact_daemon):
    """Opt-in measurement; cache eviction is advisory and never drops global caches."""
    destination = os.environ.get("EB_TEST_ARTIFACT_BENCHMARK")
    if not destination:
        pytest.skip("set EB_TEST_ARTIFACT_BENCHMARK to save measurements")
    import time
    from eb.store.ebd import EbdClient
    root, daemon = artifact_daemon
    directory = root / "state/runs/run_benchmark/outputs"
    directory.mkdir(parents=True)
    reference = {"namespace":"run","run_id":"run_benchmark","path":"outputs"}
    client = EbdClient()
    data = b"x" * 4096
    records = []
    for index in range(1000):
        name = f"file-{index:04d}"
        (directory / name).write_bytes(data)
        records.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})

    def rss(pid):
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        return 0

    def evict(paths):
        for path in paths:
            with path.open("rb") as file:
                os.fsync(file.fileno())
                os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)

    import sys
    api_socket = str(root / "benchmark-api.sock")
    api_code = r'''import asyncio, sys
from pathlib import Path
from fastapi import FastAPI
from eb.store.ebd import EbdClient
import uvicorn
app = FastAPI()
@app.post("/{kind}")
async def operation(kind: str, body: dict):
    client = EbdClient()
    if kind == "individual":
        for item in body["files"]:
            result = await client.scan(str(Path(sys.argv[2]) / item["path"]))
            assert result["samples"][0]["sha256"] == item["sha256"]
        return {"verified":True}
    if kind == "verify":
        return await client.verify_artifacts(body["root"], body["files"], max_files=1000)
    item = body["files"][0]
    if kind == "legacy-copy":
        source = Path(sys.argv[2]) / item["path"]
        payload = source.read_bytes()
        (source.parent / (source.name + ".legacy-copy")).write_bytes(payload)
        return {"copied":len(payload)}
    return await client.publish_file(body["root"], item["path"], expected_bytes=item["bytes"], expected_sha256=item["sha256"])
uvicorn.run(app, uds=sys.argv[1], log_level="error")
'''
    api_log = (root / "benchmark-api.log").open("w")
    api_process = subprocess.Popen([sys.executable,"-c",api_code,api_socket,str(directory)], stdout=api_log,stderr=api_log)
    try:
        for _ in range(100):
            if api_process.poll() is not None:
                raise RuntimeError((root / "benchmark-api.log").read_text())
            if Path(api_socket).exists(): break
            await asyncio.sleep(.05)
        else: raise RuntimeError("benchmark FastAPI did not start")
    except BaseException:
        api_process.terminate(); api_process.wait(timeout=5); api_log.close(); raise
    api = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=api_socket),base_url="http://api",timeout=3600)

    async def api_call(kind, files):
        response = await api.post("/" + kind,json={"root":reference,"files":files})
        response.raise_for_status()

    async def measure(operation):
        health = []
        peak = [rss(api_process.pid),rss(daemon.pid)]
        done = asyncio.Event()
        async def monitor():
            async with client._client(timeout=30) as transport:
                while not done.is_set():
                    begin = time.perf_counter()
                    response = await transport.get("http://ebd/healthz"); response.raise_for_status()
                    health.append((time.perf_counter()-begin)*1000)
                    peak[0] = max(peak[0],rss(api_process.pid)); peak[1] = max(peak[1],rss(daemon.pid))
                    await asyncio.sleep(.005)
        observer = asyncio.create_task(monitor())
        begin = time.perf_counter()
        try:
            await operation()
        finally:
            elapsed = time.perf_counter()-begin
            done.set(); await observer
        return {"seconds":round(elapsed,6),"api_client_peak_rss_kib":peak[0],"ebd_peak_rss_kib":peak[1],
                "health_max_ms":round(max(health, default=0),3),"health_samples":len(health)}

    try:
        measurements = []
        for count in (1,100,1000):
            files = records[:count]
            for cache in ("advisory-cold", "warm"):
                if cache == "advisory-cold":
                    evict([directory / item["path"] for item in files])
                async def individual():
                    await api_call("individual", files)
                measurements.append({"case":"manifest","files":count,"cache":cache,"method":"individual-scan", "rpc_count":count,
                    "logical_payload_read_bytes":count*len(data), **await measure(individual)})
                if cache == "advisory-cold":
                    evict([directory / item["path"] for item in files])
                measurements.append({"case":"manifest","files":count,"cache":cache,"method":"batch-verify", "rpc_count":1,
                    "logical_payload_read_bytes":count*len(data), **await measure(lambda:api_call("verify",files))})
        for mib in (64,512):
            source = directory / f"large-{mib}"
            digest = hashlib.sha256()
            block = b"a" * 1024**2
            with source.open("wb") as output:
                for _ in range(mib):
                    output.write(block); digest.update(block)
                output.flush(); os.fsync(output.fileno())
            record = {"path":source.name,"bytes":mib*1024**2,"sha256":digest.hexdigest()}
            for cache in ("advisory-cold","warm"):
                if cache == "advisory-cold": evict([source])
                measurements.append({"case":"large-file","mib":mib,"cache":cache,"method":"batch-verify","rpc_count":1,
                    "logical_payload_read_bytes":record["bytes"], **await measure(lambda:api_call("verify",[record]))})
                cas = root / "state/cas" / record["sha256"][:2] / record["sha256"]
                if cas.exists(): cas.unlink()  # Only this isolated benchmark's own unreferenced output.
                if cache == "advisory-cold": evict([source])
                measurements.append({"case":"large-file","mib":mib,"cache":cache,"method":"legacy-python-copy","rpc_count":0,
                    "logical_payload_read_bytes":record["bytes"], **await measure(lambda:api_call("legacy-copy",[record]))})
                if cache == "advisory-cold": evict([source])
                measurements.append({"case":"large-file","mib":mib,"cache":cache,"method":"publish-file","rpc_count":1,
                    "logical_payload_read_bytes":2*record["bytes"], **await measure(lambda:api_call("publish",[record]))})
        Path(destination).write_text(json.dumps({"environment":{"filesystem":subprocess.check_output(["stat","-f","-c","%T",str(root)],text=True).strip(),
            "cache":"POSIX_FADV_DONTNEED advisory only; no global cache drop", "api":"isolated FastAPI/uvicorn transport harness using production EbdClient; excludes business DB/auth",
            "comparison":"sequential per-file scan RPC baseline; historical concurrent verifiers may differ",
            "read_bytes":"logical bytes derived from operation; not physical block IO"},"measurements":measurements},indent=2))
    finally:
        await api.aclose()
        api_process.terminate()
        api_process.wait(timeout=5)
        api_log.close()
