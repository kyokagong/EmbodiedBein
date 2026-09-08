from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from eb.agent.tools import AgentCtx  # noqa: F401 — register ToolBus tools
from eb.api.routes import api_v1, health_router, media_router, spa_router, ws_router
from eb.api.embodiment_routes import router as embodiment_router
from eb.api.routes_data import router as data_router
from eb.api.routes_env import router as env_router
from eb.api.routes_model import router as model_router
from eb.api.routes_settings import router as settings_router
from eb.api.routes_train import router as train_router
from eb.bootstrap import seed_if_empty
from eb.scheduler.loop import Scheduler
from eb.settings import get_settings
from eb.store import db as store_db
from eb.store.db import init_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_engine()
    assert store_db.SessionLocal is not None
    scheduler = Scheduler()
    app.state.scheduler = scheduler
    from eb.pluginhost import PluginHost

    host = PluginHost()
    try:
        import sys

        sys.path.append(str(Path(__file__).resolve().parents[3] / "plugins"))
        from eb_plugin_isaaclab import setup as isaac_setup

        host.load("isaaclab", isaac_setup)
    except Exception:
        pass
    app.state.plugins = host
    async with store_db.SessionLocal() as session:
        await seed_if_empty(session)
        try:
            await scheduler.reconcile(session)
        except Exception:
            pass

    async def _loop() -> None:
        while True:
            await asyncio.sleep(3)
            try:
                async with store_db.SessionLocal() as session:
                    await scheduler.tick(session)
            except Exception:
                pass

    task = asyncio.create_task(_loop())
    yield
    task.cancel()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EmbodiedBein", version="0.1.0", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, same_site="lax", https_only=False)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(api_v1, prefix="/api/v1")
    app.include_router(embodiment_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(env_router, prefix="/api/v1")
    app.include_router(train_router, prefix="/api/v1")
    app.include_router(model_router, prefix="/api/v1")
    app.include_router(settings_router, prefix="/api/v1")
    app.include_router(ws_router)
    app.include_router(media_router)
    dist = settings.web_dist
    if not dist.exists():
        alt = Path(__file__).resolve().parents[2] / "web" / "dist"
        if alt.exists():
            dist = alt
    app.state.web_dist = dist
    app.include_router(spa_router)
    return app


app = create_app()
