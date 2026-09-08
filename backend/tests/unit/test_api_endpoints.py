from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from httpx import ASGITransport, AsyncClient
from eb.api.app import create_app
from eb.store.db import get_session


@pytest.fixture
def anyio_backend():
    return "asyncio"


def get_mock_session():
    mock_session = AsyncMock()
    mock_session.scalar.return_value = SimpleNamespace(principal="user:kk", actor_type="user")
    return mock_session


@pytest.mark.anyio
async def test_api_system_quotas():
    app = create_app()
    mock_session = get_mock_session()

    async def override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/system/quotas", headers={"Authorization": "Bearer eb-dev-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "disk" in data
        assert "agent_gpu_hours" in data
        assert data["concurrency"]["max_concurrent_runs"] == 4


@pytest.mark.anyio
async def test_api_system_settings():
    app = create_app()
    mock_session = get_mock_session()

    async def override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/system/settings", headers={"Authorization": "Bearer eb-dev-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "0.1.0"
        assert len(data["images"]) >= 4
        assert "scanner" in data
        assert "notifications" in data


@pytest.mark.anyio
async def test_api_preview_command():
    app = create_app()
    mock_session = get_mock_session()

    async def override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "kind": "train.rl",
            "embodiment_env_id": "ee_bhl_biped_12dof_v1",
            "params": {"task": "Velocity-Biped-v0", "num_envs": 64, "max_iterations": 5, "seed": 0, "stack": "isaaclab_rslrl"},
            "resources": {"gpus": 1, "shm_gb": 2},
            "image": "eb/isaac-lab:2.3.2",
        }
        resp = await client.post("/api/v1/jobs/preview-command", json={"spec": payload}, headers={"Authorization": "Bearer eb-dev-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "docker run" in data["command"]
        assert "eb/isaac-lab:2.3.2" in data["command"]
