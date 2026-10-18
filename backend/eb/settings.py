from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EB_", extra="ignore")

    db_url: str = "postgresql+asyncpg://eb:eb@localhost:5432/eb"
    hf_endpoint: str = "https://huggingface.co"
    redis_url: str = "redis://127.0.0.1:6380/0"
    s3_endpoint: str = "http://127.0.0.1:9000"
    s3_access_key: str = "ebadmin"
    s3_secret_key: str = "eb-minio-change-me"
    s3_bucket: str = "eb"
    data_root: Path = Path("/data/kk/embodiedbein")
    models_root: Path = Path("/data/kk/models")
    datasets_root: Path = Path("/data/kk/data")
    assets_root: Path = Path("/data/kk/assets")
    ebd_uds: str = "/run/eb/ebd.sock"
    ebd_media_uds: str = "/run/eb/ebd-http.sock"
    secret_key: str = "change-me-please-use-a-long-random-string"
    admin_user: str = "kk"
    admin_password: str = "embodied"
    agent_weekly_gpu_hours: float = 60.0
    contracts_dir: Path = Path("/app/contracts")
    ingest_roots: str = "/data/kk/Berkeley-Humanoid-Lite,/data/kk/berkeley_ego_data"
    fake_gpus: str = ""  # e.g. "0,1,2,3" for tests; empty means never invent GPUs
    bhl_repo: Path = Path("/data/kk/Berkeley-Humanoid-Lite")
    rlinf_repo: Path = Path("/data/kk/RLinf")
    ufo_native_root: Path = Path('/data/kk/UFO')
    robotwin_repo: Path = Path("/data/kk/embodiedbein/sources/robotwin-rlinf-support")
    robotwin_assets_root: Path = Path("/data/kk/data/robotwin-assets/981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042")
    web_dist: Path = Path("/app/web/dist")
    allowed_path_prefixes: str = (
        "/data/kk/embodiedbein,/data/kk/models,/data/kk/data,/data/kk/assets,"
        "/data/kk/Berkeley-Humanoid-Lite,/data/kk/RLinf,/data/kk/UFO,/data/kk/berkeley_ego_data,"
        "/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
