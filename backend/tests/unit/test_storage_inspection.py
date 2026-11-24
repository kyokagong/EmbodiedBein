from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from eb.store.ebd import EbdClient


@pytest.mark.asyncio
async def test_storage_inspection_and_text_use_bounded_uds_requests(monkeypatch):
    value = EbdClient()
    root = {"namespace": "state", "path": "runs/run_test"}
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/v1/storage/inspect":
            return httpx.Response(200, json={"schema_version": "eb/storage-inspection/v1", "mode": "inventory",
                "items": [{"path": "robot.urdf", "is_dir": False, "is_symlink": False, "bytes": 8, "mtime": 0}],
                "total": 1, "next_cursor": None})
        assert request.url.path == "/v1/storage/read-text"
        return httpx.Response(200, json={"schema_version": "eb/storage-text/v1", "text": "<robot/>"})

    monkeypatch.setattr(value, "_client", lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    result = await value.inspect_storage(root, "", mode="inventory", recursive=True, limit=1000)
    assert result["items"][0]["path"] == "robot.urdf"
    assert await value.read_storage_text(root, "robot.urdf") == "<robot/>"
    assert len(requests) == 2
    with pytest.raises(ValueError):
        await value.inspect_storage(root, "", mode="sql")


@pytest.mark.asyncio
async def test_scan_uses_inventory_and_only_reads_required_text(monkeypatch, tmp_path):
    from eb.workbench import apply_scan
    row = NS(root_path=str(tmp_path), source={}, status="scanning")
    monkeypatch.setattr("eb.store.ebd.storage_root", lambda _: {"namespace": "state", "path": "runs/run_test"})
    inventory = AsyncMock(return_value={"items": [
        {"path": "robot.urdf", "is_dir": False, "is_symlink": False},
        {"path": "mesh.stl", "is_dir": False, "is_symlink": False},
    ], "next_cursor": None, "total": 2})
    text = AsyncMock(return_value='<robot><joint name="j" type="revolute"/></robot>')
    monkeypatch.setattr(EbdClient, "inspect_storage", inventory)
    monkeypatch.setattr(EbdClient, "read_storage_text", text)
    result = await apply_scan(row)
    assert result["default_urdf"] == "robot.urdf"
    assert row.status == "ready"
    assert text.await_count == 1


@pytest.mark.asyncio
async def test_storage_hash_checks_identity(monkeypatch):
    value = EbdClient()
    monkeypatch.setattr(value, "_client", lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"schema_version": "eb/storage-hash/v1", "bytes": 8, "sha256": "a" * 64}))))
    assert (await value.hash_storage_file({"namespace": "assets"}, "robot.usd"))["bytes"] == 8


def test_urdf_collection_consumes_text_without_filesystem():
    from eb.domain.urdf import collect_urdf_joints
    assert collect_urdf_joints({"robot.urdf": '<robot><joint name="j" type="fixed"/></robot>'}) == (["j"], "robot.urdf")


@pytest.mark.asyncio
async def test_dataset_tree_uses_inventory_page(monkeypatch, tmp_path):
    from eb.api.routes_data import dataset_tree
    session = NS(get=AsyncMock(return_value=NS(dest_path=str(tmp_path), meta={})))
    monkeypatch.setattr("eb.store.ebd.storage_root", lambda _: {"namespace": "data"})
    inspect = AsyncMock(return_value={"total": 51, "items": [
        {"path": "file.txt", "is_dir": False, "is_symlink": False, "bytes": 3, "mtime": 7}]})
    monkeypatch.setattr(EbdClient, "inspect_storage", inspect)
    result = await dataset_tree("ds_test", session, NS(), path="nested", page=2, page_size=50)
    assert result["items"] == [{"name": "file.txt", "path": "nested/file.txt", "is_dir": False, "bytes": 3, "mtime": 7}]
    assert inspect.call_args.kwargs["cursor"] == "50"


def test_checkpoint_inventory_preserves_exact_members_and_pending_rejection(tmp_path):
    from eb.checkpoints import checkpoint_members
    records = [{"path": "checkpoint/model", "is_dir": False, "is_symlink": False}]
    assert checkpoint_members(tmp_path, records, 100) == {"checkpoint/model": tmp_path / "checkpoint/model"}
    for path in (".saving", "checkpoint/../other", "checkpoint/model.tmp"):
        with pytest.raises(ValueError):
            checkpoint_members(tmp_path, [{"path": path, "is_dir": False, "is_symlink": False}], 100, reject_temporary=True)


@pytest.mark.asyncio
async def test_preview_inspects_only_referenced_asset_directories(monkeypatch,tmp_path):
    from eb.api.embodiment_routes import embodiment_preview
    row=NS(root_path=str(tmp_path),source={"path":str(tmp_path)},default_urdf=None,urdf_variants=[])
    asset=NS(paths={"urdf":"robot/urdf/robot.urdf"})
    session=NS(get=AsyncMock(return_value=row),scalars=AsyncMock(return_value=NS(all=lambda:[asset])))
    monkeypatch.setattr("eb.store.ebd.storage_root",lambda _:{"namespace":"source"})
    text='<robot><joint name="j" type="fixed"/><mesh filename="../meshes/base.stl"/></robot>'
    monkeypatch.setattr(EbdClient,"read_storage_text",AsyncMock(return_value=text))
    async def inventory(root,*,recursive=False,**_):
        assert root in {tmp_path/"robot/meshes",tmp_path/"robot/usd"}, "preview must not scan the repository"
        assert recursive is False
        return [{"path":"base.stl","is_dir":False,"is_symlink":False,"bytes":4}] if root.name=="meshes" else []
    inspect=AsyncMock(side_effect=inventory)
    monkeypatch.setattr("eb.store.ebd.storage_inventory",inspect)
    result=await embodiment_preview("emb_test",session,NS(),urdf=None)
    assert result["joints"]==["j"] and result["meshes"][0]["bytes"]==4
    assert inspect.await_count==2
