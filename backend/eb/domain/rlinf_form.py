from __future__ import annotations

import binascii
import json
import re
from dataclasses import dataclass, field

RAY_BASE = 6390
RAY_SPAN = 50

FORCED = [
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
]

RLINF_ENTRYPOINTS = {
    "sft":"examples/sft/train_vla_sft.py",
    "ppo":"examples/embodiment/train_embodied_agent.py",
    "eval":"evaluations/eval_embodied_agent.py",
}


@dataclass(frozen=True)
class RlinfTemplate:
    id: str
    config_path: str
    config_name: str
    script: str
    kind: str  # ppo | sft | collect
    embodiment_env_id: str
    task_id: str
    label: str
    known_keys: frozenset[str] = field(default_factory=frozenset)
    embodiment_id: str = "emb_berkeley_humanoid_lite"
    requires_dataset: bool = False
    requires_base_model: bool = False
    default_overrides: tuple[str, ...] = ()
    model_id: str | None = None
    model_framework: str | None = None
    ready: bool = True
    dependency_ready: bool = True
    blocked_reason: str | None = None
    native_embodiments: tuple[str, ...] = ()


def template_entrypoint(template: RlinfTemplate) -> str:
    return RLINF_ENTRYPOINTS.get(template.kind, template.script)


TEMPLATES: tuple[RlinfTemplate, ...] = (
    RlinfTemplate(
        id="rpo_pbfm_fb_depth_native", config_path="pbfm/configs/robots", config_name="rpo_23dof",
        script="eb.runners.rlinf_pbfm_entry", kind="native_rl",
        embodiment_env_id="ee_rpo_v2_pbfm_depth_23dof_v1", task_id="tsk_rpo_pbfm_terrain_locomotion_v1",
        label="RPO · P-BFM 深度地形 FB-CPR-Aux", embodiment_id="emb_roboparty_rpo",
        requires_dataset=True, model_framework="pbfm_native_fb_depth",
        ready=False, dependency_ready=False,
        blocked_reason="P-BFM/RPO 组合镜像、数据 loader 与本体/相机闭环证据尚未完成；可预览诊断，禁止启动训练。",
    ),
    RlinfTemplate(
        id="rpo_ufo_fb_native", config_path="configs/robots", config_name="rpo_23dof",
        script="eb.runners.rlinf_ufo_entry", kind="native_rl",
        embodiment_env_id="ee_rpo_v2_ufo_native_23dof_v2", task_id="tsk_rpo_ufo_native_stand_walk_v2",
        label="RPO · UFO 原生 BFM-Zero（FB-CPR-Aux）", embodiment_id="emb_roboparty_rpo",
        requires_dataset=True, model_framework="ufo_native_fb_cpr_aux",
    ),
    RlinfTemplate(
        id="rpo_v2_flat_ppo", config_path="../eb_runner/config", config_name="rpo_flat_ppo",
        script="examples/embodiment/train_embodied_agent.py", kind="ppo",
        embodiment_env_id="ee_rpo_v2_flat_23dof_v1", task_id="tsk_rpo_flat",
        label="RPO-Flat 官方速度指令＋转向 PPO", embodiment_id="emb_roboparty_rpo",
        default_overrides=("runner.max_epochs=500", "runner.save_interval=25",
            "runner.val_check_interval=25", "env.train.total_num_envs=512",
            "env.eval.total_num_envs=64", "actor.micro_batch_size=768",
            "actor.global_batch_size=3072", "actor.optim.lr=1e-3",
            "actor.optim.value_lr=1e-3", "actor.seed=42", "env.train.rollout_epoch=1"),
    ),
    RlinfTemplate(
        id="rpo_v2_flat_eval", config_path="../eb_runner/config", config_name="rpo_flat_ppo",
        script="evaluations/eval_embodied_agent.py", kind="eval",
        embodiment_env_id="ee_rpo_v2_flat_23dof_v1", task_id="tsk_rpo_flat",
        label="RPO-Flat 官方指令分布 Isaac ONNX 独立评估",
        embodiment_id="emb_roboparty_rpo", requires_base_model=True,
    ),
    RlinfTemplate(
        id="rpo_v2_amp_stand_walk_ppo", config_path="../eb_runner/config",
        config_name="rpo_amp_stand_walk_ppo",
        script="examples/embodiment/train_embodied_agent.py", kind="ppo",
        embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
        task_id="tsk_rpo_amp_stand_walk", label="RPO V2 AMP 直立＋自然直行",
        embodiment_id="emb_roboparty_rpo", requires_dataset=True,
        known_keys=frozenset({"runner.max_epochs", "env.train.total_num_envs", "actor.optim.lr"}),
        default_overrides=("runner.max_epochs=1000", "runner.save_interval=50",
            "runner.val_check_interval=25", "env.train.total_num_envs=512",
            "env.eval.total_num_envs=64", "actor.micro_batch_size=1024",
            "actor.global_batch_size=4096", "actor.optim.lr=3e-4",
            "actor.optim.value_lr=3e-4", "actor.seed=1234", "env.train.rollout_epoch=1"),
    ),
    RlinfTemplate(
        id="rpo_v2_amp_eval", config_path="../eb_runner/config",
        config_name="rpo_amp_stand_walk_ppo", script="evaluations/eval_embodied_agent.py",
        kind="eval", embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
        task_id="tsk_rpo_amp_stand_walk", label="RPO V2 AMP Isaac ONNX 闭环诊断",
        embodiment_id="emb_roboparty_rpo", requires_base_model=True,
        default_overrides=("runner.only_eval=True", "env.eval.total_num_envs=1"),
    ),
    RlinfTemplate(
        id="rpo_v2_wbc_stand_walk_ppo", config_path="../eb_runner/config",
        config_name="rpo_wbc_stand_walk_ppo",
        script="examples/embodiment/train_embodied_agent.py", kind="ppo",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", task_id="tsk_rpo_stand_walk",
        label="RPO V2 直立＋低速直行 PPO", embodiment_id="emb_roboparty_rpo",
        known_keys=frozenset({"runner.max_epochs", "env.train.total_num_envs", "actor.optim.lr"}),
        default_overrides=(
            "runner.max_epochs=1000", "runner.save_interval=25", "runner.val_check_interval=25",
            "env.train.total_num_envs=512", "env.eval.total_num_envs=64",
            "actor.micro_batch_size=128", "actor.global_batch_size=1024",
            "actor.optim.lr=3e-4", "actor.optim.value_lr=3e-4",
            "actor.seed=1234", "env.train.rollout_epoch=1",
        ),
    ),
    RlinfTemplate(
        id="rpo_v2_wbc_ppo",
        config_path="../eb_runner/config",
        config_name="rpo_wbc_ppo",
        script="examples/embodiment/train_embodied_agent.py",
        kind="ppo",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_walk_upperbody_tracking",
        label="RPO V2 行走＋上半身 WBC PPO",
        embodiment_id="emb_roboparty_rpo",
        known_keys=frozenset({"runner.max_epochs", "env.train.total_num_envs", "actor.optim.lr"}),
        default_overrides=(
            "runner.max_epochs=1", "runner.save_interval=25",
            "runner.val_check_interval=25", "env.train.total_num_envs=512",
            "env.eval.total_num_envs=64", "actor.micro_batch_size=128",
            "actor.global_batch_size=1024", "actor.optim.lr=3e-4",
            "actor.optim.value_lr=3e-4", "actor.seed=1234", "env.train.rollout_epoch=1",
        ),
    ),
    RlinfTemplate(
        id="rpo_v2_wbc_eval", config_path="../eb_runner/config",
        config_name="rpo_wbc_ppo", script="evaluations/eval_embodied_agent.py",
        kind="eval", embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_walk_upperbody_tracking", label="RPO V2 WBC Isaac 闭环评估",
        embodiment_id="emb_roboparty_rpo", requires_base_model=True,
        default_overrides=("runner.only_eval=True", "env.eval.total_num_envs=64",
                           "env.eval.video_cfg.save_video=False"),
    ),
    RlinfTemplate(
        id="berkeley_wbc_ppo_mlp",
        config_path="examples/embodiment/config",
        config_name="berkeley_wbc_ppo_mlp",
        script="examples/embodiment/train_embodied_agent.py",
        kind="ppo",
        embodiment_env_id="ee_bhl_wbc_22dof_v1",
        task_id="tsk_bhl_wbc_tracking",
        label="全身控制 PPO",
        known_keys=frozenset({"runner.max_epochs", "env.train.total_num_envs", "actor.optim.lr"}),
    ),
    RlinfTemplate(
        id="berkeley_reach_sft",
        config_path="examples/sft/config",
        config_name="berkeley_reach_sft_openpi_pi05",
        script="examples/sft/train_vla_sft.py",
        kind="sft",
        embodiment_env_id="ee_bhl_fixedhand_10dof_v1",
        task_id="tsk_bhl_reach",
        label="Reach SFT",
        known_keys=frozenset({"runner.max_epochs", "actor.optim.lr"}),
    ),
    RlinfTemplate(
        id="collect_berkeley_reach",
        config_path="examples/embodiment/config",
        config_name="collect_berkeley_reach",
        script="examples/embodiment/collect_berkeley_reach.py",
        kind="collect",
        embodiment_env_id="ee_bhl_fixedhand_10dof_v1",
        task_id="tsk_bhl_reach",
        label="Reach 采集",
        known_keys=frozenset({"env.train.total_num_envs"}),
    ),
)


TEMPLATES += (
    RlinfTemplate(
        id="libero_pi05_sft", config_path="examples/sft/config",
        config_name="libero_sft_openpi", script="examples/sft/train_vla_sft.py",
        kind="sft", embodiment_env_id="ee_libero_franka_delta_v1",
        task_id="tsk_libero_spatial", label="LIBERO π0.5 SFT",
        embodiment_id="emb_libero_franka", requires_dataset=True, requires_base_model=True,
        default_overrides=("actor.model.openpi.config_name=pi05_libero", "actor.model.openpi.train_expert_only=True", "actor.micro_batch_size=1",
            "actor.global_batch_size=4", "actor.model.num_action_chunks=10", "actor.model.num_steps=5",
            "actor.fsdp_config.sharding_strategy=full_shard", "runner.max_steps=100",
            "runner.save_interval=100"),
    ),
    *(
        RlinfTemplate(
            id=f"libero_{suite}_pi05_{mode}", config_path="examples/embodiment/config",
            config_name=f"libero_{suite}_ppo_openpi_pi05",
            script=RLINF_ENTRYPOINTS[mode], kind=mode,
            embodiment_env_id="ee_libero_franka_delta_v1", task_id=f"tsk_libero_{suite}",
            label=f"LIBERO {suite} π0.5 {'PPO' if mode == 'ppo' else '独立评估'}",
            embodiment_id="emb_libero_franka", requires_base_model=True,
            default_overrides=("actor.micro_batch_size=1", "actor.global_batch_size=4",
                "actor.fsdp_config.sharding_strategy=full_shard", "actor.enable_offload=True",
                "rollout.enable_offload=True", "env.train.total_num_envs=4",
                "env.train.rollout_epoch=1", "env.eval.total_num_envs=4",
                # Each save writes model + optimizer shards; retain upstream's
                # periodic cadence. RLinf also saves at the final training step.
                "runner.max_epochs=1", "runner.save_interval=40",
                f"runner.only_eval={'True' if mode == 'eval' else 'False'}"),
        ) for suite in ("spatial", "object", "goal", "10") for mode in ("ppo", "eval")
    ),
)


TEMPLATES += (
    RlinfTemplate(
        id="robotwin_pi05_eval", config_path="evaluations/robotwin",
        config_name="robotwin_adjust_bottle_openpi_pi05_eval",
        script=RLINF_ENTRYPOINTS["eval"], kind="eval",
        embodiment_env_id="ee_robotwin_aloha_joint14_v1", task_id="tsk_robotwin_multitask",
        label="RoboTwin ALOHA · π0.5 闭环评估", embodiment_id="emb_robotwin_aloha_agilex",
        requires_base_model=True, model_id="pi05", model_framework="rlinf_openpi",
        native_embodiments=("aloha-agilex",),
        default_overrides=("runner.only_eval=True", "rollout.model.openpi.config_name=pi05_aloha_robotwin",
            "runner.enable_decoupled_mode=False", "rollout.enable_offload=False",
            "env.eval.video_cfg.save_video=False",
            "rollout.model.add_value_head=False",
            "rollout.model.action_dim=14", "rollout.model.num_action_chunks=50",
            "rollout.model.openpi.num_images_in_input=3"),
    ),
    RlinfTemplate(
        id="robotwin_lingbot_vla2_eval", config_path="../eb_runner/config",
        config_name="robotwin_eval_lingbot_vla2", script=RLINF_ENTRYPOINTS["eval"],
        kind="eval", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · LingBot-VLA-2 闭环评估",
        embodiment_id="emb_robotwin_aloha_agilex", requires_base_model=True,
        model_id="lingbot-vla-2", model_framework="lingbot_vla2",
        ready=False, blocked_reason="policy_loader_missing: LingBot-VLA-2 closed-loop bridge 尚未验证",
        native_embodiments=("aloha-agilex",)),
    RlinfTemplate(
        id="robotwin_gr00t_n17_eval", config_path="../eb_runner/config",
        config_name="robotwin_eval_gr00t_n17", script=RLINF_ENTRYPOINTS["eval"],
        kind="eval", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · GR00T-N1.7 闭环评估",
        embodiment_id="emb_robotwin_aloha_agilex", requires_base_model=True,
        model_id="gr00t-n1.7", model_framework="gr00t_n1d7",
        ready=False, blocked_reason="policy_loader_missing: GR00T-N1.7 closed-loop bridge 尚未验证",
        native_embodiments=("aloha-agilex",)),
    RlinfTemplate(
        id="robotwin_dm05_eval", config_path="../eb_runner/config",
        config_name="robotwin_eval_dm05", script=RLINF_ENTRYPOINTS["eval"],
        kind="eval", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · 原力灵机 DM0.5 闭环评估",
        embodiment_id="emb_robotwin_aloha_agilex", requires_base_model=True,
        model_id="dm05", model_framework="opendm_dm05",
        ready=False, blocked_reason="policy_loader_missing: DM0.5 closed-loop bridge 尚未验证",
        native_embodiments=("aloha-agilex",)),
    RlinfTemplate(
        id="robotwin_xr1_eval", config_path="../eb_runner/config",
        config_name="robotwin_eval_xiaomi_xr1", script=RLINF_ENTRYPOINTS["eval"],
        kind="eval", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · Xiaomi-Robotics-1 闭环评估",
        embodiment_id="emb_robotwin_aloha_agilex", requires_base_model=True,
        model_id="xiaomi-robotics-1", model_framework="xiaomi_xr1",
        ready=False, blocked_reason="controller_bridge_unverified: 60D EEF controller/IK 尚未验证",
        native_embodiments=("aloha-agilex",)),
    RlinfTemplate(
        id="robotwin_g0_eval", config_path="../eb_runner/config",
        config_name="robotwin_eval_g0", script=RLINF_ENTRYPOINTS["eval"],
        kind="eval", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · 星海图 G0 闭环评估",
        embodiment_id="emb_robotwin_aloha_agilex", requires_base_model=True,
        model_id="galaxea-g0", model_framework="galaxea_g0",
        ready=False, blocked_reason="model_recipe_blocked: G0 RoboTwin action profile 尚未验证"),
)


TEMPLATES += (
    RlinfTemplate(
        id="robotwin_pi05_sft", config_path="examples/sft/config",
        config_name="robotwin_sft_openpi_pi05", script="examples/sft/train_vla_sft.py",
        kind="sft", embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        task_id="tsk_robotwin_multitask", label="RoboTwin ALOHA · π0.5 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True,
        requires_base_model=True, model_id="pi05", model_framework="rlinf_openpi", native_embodiments=("aloha-agilex",),
        default_overrides=("actor.model.openpi.config_name=pi05_aloha_robotwin",
            "actor.model.action_dim=14", "actor.model.num_action_chunks=50",
            "actor.model.openpi.num_images_in_input=3", "actor.model.openpi.train_expert_only=False",
            "actor.micro_batch_size=1", "actor.global_batch_size=4",
            "actor.fsdp_config.sharding_strategy=full_shard", "runner.max_steps=100",
            "runner.save_interval=100"),
    ),
    RlinfTemplate("robotwin_lingbot_vla2_sft", "../eb_runner/config", "robotwin_sft_lingbot_vla2",
        "examples/sft/train_vla_sft.py", "sft", "ee_robotwin_aloha_joint14_v1",
        "tsk_robotwin_multitask", "RoboTwin ALOHA · LingBot-VLA-2 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True, requires_base_model=True,
        model_id="lingbot-vla-2", model_framework="lingbot_vla2", native_embodiments=("aloha-agilex",),
        default_overrides=("actor.micro_batch_size=1", "actor.global_batch_size=4",
            "runner.max_steps=100", "runner.save_interval=100")),
    RlinfTemplate("robotwin_gr00t_n17_sft", "../eb_runner/config", "robotwin_sft_gr00t_n17",
        "examples/sft/train_vla_sft.py", "sft", "ee_robotwin_aloha_joint14_v1",
        "tsk_robotwin_multitask", "RoboTwin ALOHA · GR00T-N1.7 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True, requires_base_model=True,
        model_id="gr00t-n1.7", model_framework="gr00t_n1d7", native_embodiments=("aloha-agilex",),
        default_overrides=("actor.micro_batch_size=1", "actor.global_batch_size=4",
            "runner.max_steps=100", "runner.save_interval=100")),
    RlinfTemplate("robotwin_g0_sft", "examples/sft/config", "robotwin_sft_g0",
        "examples/sft/train_vla_sft.py", "sft", "ee_robotwin_aloha_joint14_v1",
        "tsk_robotwin_multitask", "RoboTwin ALOHA · 星海图 G0 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True, requires_base_model=True,
        model_id="galaxea-g0", model_framework="galaxea_g0", ready=False,
        blocked_reason="G0 尚无公开 RoboTwin action profile；不能借用 G0.5"),
    RlinfTemplate("robotwin_dm05_sft", "../eb_runner/config", "robotwin_sft_dm05",
        "examples/sft/train_vla_sft.py", "sft", "ee_robotwin_aloha_joint14_v1",
        "tsk_robotwin_multitask", "RoboTwin ALOHA · 原力灵机 DM0.5 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True, requires_base_model=True,
        model_id="dm05", model_framework="opendm_dm05", native_embodiments=("aloha-agilex",),
        default_overrides=("actor.micro_batch_size=1", "actor.global_batch_size=4",
            "runner.max_steps=100", "runner.save_interval=100")),
    RlinfTemplate("robotwin_xr1_sft", "../eb_runner/config", "robotwin_sft_xiaomi_xr1",
        "examples/sft/train_vla_sft.py", "sft", "ee_robotwin_aloha_joint14_v1",
        "tsk_robotwin_multitask", "RoboTwin ALOHA · Xiaomi-Robotics-1 SFT",
        embodiment_id="emb_robotwin_aloha_agilex", requires_dataset=True, requires_base_model=True,
        model_id="xiaomi-robotics-1", model_framework="xiaomi_xr1",
        native_embodiments=("aloha-agilex",),
        default_overrides=("actor.micro_batch_size=1", "actor.global_batch_size=4",
            "runner.max_steps=100", "runner.save_interval=100")),
)


def get_template(template_id: str) -> RlinfTemplate:
    for t in TEMPLATES:
        if t.id == template_id:
            return t
    raise KeyError(template_id)


def require_template_ready(template: RlinfTemplate) -> None:
    if not template.ready or not template.dependency_ready:
        raise ValueError(f"RLInf 模板尚未就绪：{template.blocked_reason or template.id}")


def template_image(template: RlinfTemplate) -> str:
    if template.id == "rpo_pbfm_fb_depth_native":
        return "eb/rlinf-pbfm:rpo-depth-v1"
    if template.id == "rpo_ufo_fb_native":
        from eb.ufo_training import IMAGE
        return IMAGE
    if template.id.startswith(("rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_")):
        return "eb/rlinf:rpo-v2-wbc"
    if template.id == "robotwin_pi05_eval":
        return "eb/rlinf:robotwin-pi05"
    if template.id in {"robotwin_lingbot_vla2_sft", "robotwin_lingbot_vla2_eval"}:
        return "eb/rlinf:robotwin-lingbot-vla2"
    if template.id in {"robotwin_dm05_sft", "robotwin_dm05_eval"}:
        return "eb/rlinf:robotwin-dm05"
    if template.id in {"robotwin_gr00t_n17_sft", "robotwin_gr00t_n17_eval"}:
        return "eb/rlinf:robotwin-gr00t-n1.7"
    if template.id in {"robotwin_xr1_sft", "robotwin_xr1_eval"}:
        return "eb/rlinf:robotwin-xiaomi-xr1"
    return "eb/rlinf:libero-pi05" if template.id.startswith(("libero_", "robotwin_")) else "eb/rlinf:0.4.0-emb"


def component_placement(template_id: str, gpus: int) -> str:
    template = get_template(template_id)
    if not 1 <= gpus <= 4:
        raise ValueError("GPU 数必须为 1–4")
    if template_id.startswith("libero_") and template.kind == "ppo":
        # One rollout replica avoids retaining a full CPU policy on every actor rank.
        return "+cluster.component_placement={actor:0-" + str(gpus - 1) + ",env\\,\\ rollout:0}"
    if template_id.startswith("robotwin_") and template.kind == "eval":
        env_range = "0" if gpus == 1 else f"0-{gpus - 1}"
        return f"cluster.component_placement={{env:{env_range},rollout:0}}"
    if template.kind == "eval":
        return "cluster.component_placement={env\\,\\ rollout:0-" + str(gpus - 1) + "}"
    return "cluster.component_placement={actor\\,\\ env\\,\\ rollout:0-" + str(gpus - 1) + "}"


def ray_port_for(job_id: str) -> int:
    crc = binascii.crc32(job_id.encode()) & 0xFFFFFFFF
    return RAY_BASE + (crc % RAY_SPAN)


def forced_overrides(job_name: str) -> list[str]:
    return [
        *FORCED,
        "runner.logger.experiment_name=" + json.dumps(job_name, ensure_ascii=False),
    ]


# Restrict user overrides to scalar tuning knobs; input/runtime/interface bindings are owned by the platform.
TUNABLE = frozenset({"runner.max_epochs", "runner.max_steps", "runner.save_interval", "runner.val_check_interval",
    "env.train.total_num_envs", "env.eval.total_num_envs", "env.train.rollout_epoch", "env.eval.rollout_epoch",
    "actor.optim.lr", "actor.optim.value_lr", "actor.micro_batch_size", "actor.global_batch_size", "actor.seed",
    "amp_discriminator.style_weight_max"})


def validate_hydra_overrides(lines: list[str]) -> list[str]:
    errors = []
    for line in lines:
        if not line.strip():
            continue
        key, sep, value = line.strip().partition("=")
        if not sep or key not in TUNABLE or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value):
            errors.append(f"不允许的覆盖（仅支持训练数值参数）: {line}")
    return errors


def build_overrides(template_id: str, gpus: int, job_name: str, extra: list[str]) -> list[str]:
    if get_template(template_id).kind == "native_rl":
        if extra:
            raise ValueError("原生训练使用 recipe 专属冻结参数，不接受 Hydra overrides")
        return []
    errors = validate_hydra_overrides(extra)
    if errors:
        raise ValueError("; ".join(errors))
    style = [item.split("=", 1)[1] for item in extra
             if item.startswith("amp_discriminator.style_weight_max=")]
    if style and (template_id != "rpo_v2_amp_stand_walk_ppo" or len(style) != 1 or float(style[0]) != 0):
        raise ValueError("AMP 风格权重仅允许在同接口 PPO 对照中固定为 0")
    # One value per key: Hydra errors on duplicate defaults selections.
    values = dict(item.split("=", 1) for item in get_template(template_id).default_overrides)
    values.update(item.strip().split("=", 1) for item in extra if item.strip())
    if template_id in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo"}:
        try:
            micro = int(values["actor.micro_batch_size"])
            global_batch = int(values["actor.global_batch_size"])
            save_interval = int(values["runner.save_interval"])
            val_interval = int(values["runner.val_check_interval"])
        except (KeyError, ValueError) as exc:
            raise ValueError("RPO batch size 和保存/评估间隔必须是整数") from exc
        if micro <= 0 or global_batch <= 0 or global_batch % (micro * gpus):
            raise ValueError("global_batch_size 必须能被 micro_batch_size × GPU 数整除")
        if val_interval > 0 and save_interval >= 0 and save_interval % val_interval:
            raise ValueError("runner.save_interval 必须能被 runner.val_check_interval 整除")
        if template_id == "rpo_v2_flat_ppo":
            envs = int(values["env.train.total_num_envs"])
            rollout = int(values["env.train.rollout_epoch"])
            if envs <= 0 or envs % gpus or rollout != 1 or envs * 24 != global_batch * 4:
                raise ValueError("RPO-Flat 每轮 24 步 rollout 必须恰好拆成 4 个 global batch")
        if template_id == "rpo_v2_amp_stand_walk_ppo":
            if gpus != 1:
                raise ValueError("AMP 首期只支持单 learner 单 GPU；多卡可并行独立 seed")
            envs = int(values["env.train.total_num_envs"])
            if envs <= 0 or (envs * 24) % global_batch:
                raise ValueError("AMP 单次 rollout 的 transition 数必须能被 global batch 整除")
    placement = component_placement(template_id, gpus)
    # ebd passes argv directly to Docker, so Hydra receives one escaping layer.
    placement = placement.replace("\\\\", "\\")
    # Hydra merges dictionaries, so remove the old combined-role mapping first.
    reset_placement = ["~cluster.component_placement"] if placement.startswith("+") else []
    return [f"{key}={value}" for key, value in values.items()] + reset_placement + [placement, *forced_overrides(job_name)]
