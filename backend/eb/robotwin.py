from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ROBOTWIN_HF_REPO = "TianxingChen/RoboTwin2.0"
ROBOTWIN_HF_REVISION = "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"

ROBOTWIN_TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)


@dataclass(frozen=True)
class EmbodimentProfile:
    id: str
    label: str
    embodiment_id: str
    embodiment_env_id: str
    archive_name: str
    download_ready: bool = False
    blocked_reason: str | None = "等待固定 asset/config 与样本 probe 后生成本体环境契约"


ROBOTWIN_EMBODIMENTS = {
    "aloha-agilex": EmbodimentProfile("aloha-agilex", "ALOHA / AgileX 双臂", "emb_robotwin_aloha_agilex", "ee_robotwin_aloha_joint14_v1", "aloha-agilex", True, None),
    "piper": EmbodimentProfile("piper", "Piper 双臂", "emb_robotwin_piper", "ee_robotwin_piper_native_v1", "piper"),
    "franka-panda": EmbodimentProfile("franka-panda", "Franka-Panda 双臂", "emb_robotwin_franka_panda", "ee_robotwin_franka_native_v1", "franka"),
    "ARX-X5": EmbodimentProfile("ARX-X5", "ARX-X5 双臂", "emb_robotwin_arx_x5", "ee_robotwin_arx_x5_native_v1", "arx-x5"),
    "ur5-wsg": EmbodimentProfile("ur5-wsg", "UR5-WSG 双臂", "emb_robotwin_ur5_wsg", "ee_robotwin_ur5_wsg_native_v1", "ur5"),
}


@dataclass(frozen=True)
class DownloadProfile:
    id: str
    label: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    native_embodiment: str
    embodiment_env_id: str
    scenario: str
    format: str
    requires_probe: bool = False
    task: str | None = None

    def public(self) -> dict:
        return asdict(self)


DOWNLOAD_PROFILES = {
    "aloha_lerobot_v21": DownloadProfile(
        "aloha_lerobot_v21", "ALOHA · LeRobot v2.1（π0.5 首期）", ROBOTWIN_HF_REPO,
        ROBOTWIN_HF_REVISION, ("lerobot_dataset/RoboTwin_lerobot_v21.zip",),
        "aloha-agilex", "ee_robotwin_aloha_joint14_v1", "clean", "lerobot_v2.1", True,
    ),
    "aloha_lerobot_v30": DownloadProfile(
        "aloha_lerobot_v30", "ALOHA · LeRobot v3.0", ROBOTWIN_HF_REPO,
        ROBOTWIN_HF_REVISION, ("lerobot_dataset/RoboTwin_lerobot_v30.zip",),
        "aloha-agilex", "ee_robotwin_aloha_joint14_v1", "clean", "lerobot_v3.0", True,
    ),
}


def get_download_profile(profile_id: str) -> DownloadProfile:
    try:
        return DOWNLOAD_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"未知 RoboTwin 下载配置: {profile_id}") from exc


def raw_archive_profile(task: str, scenario: str, embodiment: str) -> DownloadProfile:
    if task not in ROBOTWIN_TASKS:
        raise ValueError("未知 RoboTwin 任务")
    if scenario not in {"clean", "randomized"}:
        raise ValueError("RoboTwin 场景必须是 clean 或 randomized")
    try:
        body = ROBOTWIN_EMBODIMENTS[embodiment]
    except KeyError as exc:
        raise ValueError("未知 RoboTwin 原生本体") from exc
    count = 50 if scenario == "clean" else 500
    filename = f"{body.archive_name}_{scenario}_{count}.zip"
    return DownloadProfile(
        f"raw:{task}:{scenario}:{embodiment}", f"{task} · {body.label} · {scenario}",
        ROBOTWIN_HF_REPO, ROBOTWIN_HF_REVISION, (f"dataset/{task}/{filename}",),
        embodiment, body.embodiment_env_id, scenario, "xpolicylab_hdf5", task=task,
    )


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    template_id: str
    cross_embodiment: bool | Literal["limited"]
    direct_embodiments: tuple[str, ...]
    ready: bool
    blocked_reason: str | None = None

    def public(self) -> dict:
        return asdict(self)


ROBOTWIN_MODELS = {
    "lingbot-vla-2": ModelProfile("lingbot-vla-2", "LingBot-VLA-2", "robotwin_lingbot_vla2_sft", True, ("aloha-agilex",), True),
    "pi05": ModelProfile("pi05", "π0.5", "robotwin_pi05_sft", True, ("aloha-agilex",), True),
    "gr00t-n1.7": ModelProfile("gr00t-n1.7", "GR00T-N1.7", "robotwin_gr00t_n17_sft", True, ("aloha-agilex",), True),
    "galaxea-g0": ModelProfile("galaxea-g0", "星海图 G0", "robotwin_g0_sft", "limited", (), False, "G0 尚无公开 RoboTwin action profile，且不能借用 G0.5 profile"),
    "dm05": ModelProfile("dm05", "原力灵机 DM0.5", "robotwin_dm05_sft", True, ("aloha-agilex",), True),
    "xiaomi-robotics-1": ModelProfile("xiaomi-robotics-1", "Xiaomi-Robotics-1", "robotwin_xr1_sft", True, ("aloha-agilex",), True),
}
