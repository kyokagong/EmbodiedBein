from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["git_external", "git_snapshot", "managed_bundle", "in_image", "plugin"]


class EmbodimentSource(BaseModel):
    kind: SourceKind
    path: str | None = None
    writable: bool = False
    default_ref: str = "HEAD"
    git_url: str | None = None
    image: str | None = None
    image_path: str | None = None
    plugin: str | None = None


class Embodiment(BaseModel):
    id: str
    display_name: str
    source: EmbodimentSource
    status: str = "active"
    seed: bool = False
    tags: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)


BHL_SEED = Embodiment(
    id="emb_berkeley_humanoid_lite",
    display_name="Berkeley Humanoid Lite",
    source=EmbodimentSource(
        kind="git_external",
        path="/data/kk/Berkeley-Humanoid-Lite",
        writable=False,
        default_ref="HEAD",
    ),
    status="active",
    seed=True,
    tags=["bhl", "loco", "manip"],
    capabilities={"preview": True, "train": True, "eval": True, "deploy": True},
)

BHL_ASSET_INDEX: list[dict[str, Any]] = [
    {
        "format": "URDF",
        "path": "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf",
        "show": "3d",
    },
    {
        "format": "URDF-biped",
        "path": "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite/urdf/berkeley_humanoid_lite_biped.urdf",
        "show": "3d",
    },
]
