from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eb.ids import sha256_text


class TaskSpec(BaseModel):
    id: str
    compatible_env_ids: list[str]
    stack_bindings: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"

    def spec_hash(self) -> str:
        return sha256_text(self.model_dump_json())

    def gym_id(self, stack: str = "isaaclab_rslrl") -> str | None:
        bind = self.stack_bindings.get(stack) or {}
        return bind.get("gym_id")

    def rlinf_config_name(self) -> str | None:
        bind = self.stack_bindings.get("rlinf") or {}
        return bind.get("config_name")


BHL_SEED_TASKS: list[TaskSpec] = [
    TaskSpec(
        id="tsk_bhl_velocity_humanoid",
        status="active",
        compatible_env_ids=["ee_bhl_humanoid_22dof_v1"],
        stack_bindings={"isaaclab_rslrl": {"gym_id": "Velocity-Berkeley-Humanoid-Lite-v0"}},
    ),
    TaskSpec(
        id="tsk_bhl_velocity_biped",
        status="active",
        compatible_env_ids=["ee_bhl_biped_12dof_v1"],
        stack_bindings={"isaaclab_rslrl": {"gym_id": "Velocity-Berkeley-Humanoid-Lite-Biped-v0"}},
    ),
    TaskSpec(
        id="tsk_bhl_wbc_tracking",
        status="active",
        compatible_env_ids=["ee_bhl_wbc_22dof_v1"],
        stack_bindings={
            "isaaclab_rslrl": {"gym_id": "Tracking-Berkeley-Humanoid-Lite-WholeBody-v0"},
            "rlinf": {"config_name": "berkeley_wbc_ppo_mlp"},
        },
    ),
    TaskSpec(
        id="tsk_bhl_reach",
        status="active",
        compatible_env_ids=["ee_bhl_fixedhand_10dof_v1"],
        stack_bindings={
            "isaaclab_rslrl": {"gym_id": "Isaac-Berkeley-Humanoid-Lite-FixedHand-Reach-v0"},
            "rlinf": {"config_name": "isaaclab_berkeley_reach"},
        },
    ),
    TaskSpec(
        id="tsk_bhl_push",
        status="active",
        compatible_env_ids=["ee_bhl_fixedhand_10dof_v1"],
        stack_bindings={"isaaclab_rslrl": {"gym_id": "Isaac-Berkeley-Humanoid-Lite-FixedHand-Push-v0"}},
    ),
]
