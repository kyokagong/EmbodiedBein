from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EB_", extra="ignore")

    db_url: str = "postgresql+asyncpg://eb:eb@localhost:5432/eb"
    redis_url: str = "redis://127.0.0.1:6380/0"
    s3_endpoint: str = "http://127.0.0.1:9000"
    s3_access_key: str = "ebadmin"
    s3_secret_key: str = "eb-minio-change-me"
    s3_bucket: str = "eb"
    data_root: Path = Path("/var/lib/eb")
    ebd_uds: str = "/run/eb/ebd.sock"
    ebd_media_uds: str = "/run/eb/ebd-http.sock"
    secret_key: str = "change-me-please-use-a-long-random-string"
    admin_user: str = "kk"
    admin_password: str = "embodied"
    agent_weekly_gpu_hours: float = 60.0
    contracts_dir: Path = Path("/app/contracts")
    ingest_roots: str = "/data/kk/Berkeley-Humanoid-Lite,/data/kk/berkeley_ego_data"
    web_dist: Path = Path("/app/web/dist")


@lru_cache
def get_settings() -> Settings:
    return Settings()
