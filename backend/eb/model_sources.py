"""Trusted public checkpoints that the platform can acquire."""

MODEL_SOURCES = {
    "pi05-libero": {
        "repo_id": "RLinf/RLinf-Pi05-LIBERO-SFT",
        "label": "RLinf π0.5 LIBERO SFT",
        "embodiment_id": "emb_libero_franka",
        "embodiment_env_id": "ee_libero_franka_delta_v1",
        "framework": "rlinf_openpi",
        "allow_patterns": ("*.safetensors", "model.safetensors.index.json", "config.json", "assets/**", "physical-intelligence/**"),
        "required_files": ("physical-intelligence/libero/norm_stats.json",),
    },
    "pi05-robotwin": {
        "repo_id": "RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
        "label": "RLinf π0.5 RoboTwin ALOHA SFT · adjust_bottle",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "rlinf_openpi",
        "allow_patterns": ("*.safetensors", "model.safetensors.index.json", "config.json", "assets/**", "physical-intelligence/**"),
        "required_files": ("physical-intelligence/robotwin/norm_stats.json",),
    },
    "lingbot-vla-2": {
        "repo_id": "robbyant/lingbot-vla-v2-6b",
        "label": "LingBot-VLA-2 6B",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "lingbot_vla2",
        "allow_patterns": ("*.json", "*.safetensors", "*.model", "depth/**", "dino_video/**"),
        "required_files": ("config.json", "model.safetensors.index.json", "depth/model.pt", "dino_video/teacher_step_10000.pth"),
    },
    "moge-2-vitb-normal": {
        "repo_id": "Ruicheng/moge-2-vitb-normal",
        "label": "MoGe-2 ViT-B Normal (LingBot depth teacher)",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "lingbot_depth_teacher",
        "allow_patterns": ("model.pt", "README.md"),
        "required_files": ("model.pt",),
    },
    "gr00t-n1.7": {
        "repo_id": "nvidia/GR00T-N1.7-3B",
        "label": "NVIDIA GR00T-N1.7 3B",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "gr00t_n1d7",
        "allow_patterns": ("*",),
        "required_files": ("config.json", "model.safetensors.index.json", "processor_config.json", "statistics.json", "embodiment_id.json"),
    },
    "cosmos-reason2-2b": {
        "repo_id": "nvidia/Cosmos-Reason2-2B",
        "label": "Cosmos-Reason2-2B (GR00T backbone)",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "qwen3_vl_backbone",
        "allow_patterns": ("*.json", "*.safetensors", "*.txt"),
        "required_files": ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"),
    },
    "galaxea-g0": {
        "repo_id": "OpenGalaxea/G0-VLA",
        "label": "OpenGalaxea G0 3B",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "galaxea_g0",
        "allow_patterns": ("G0_3B_base.pt", "G0_3B_base_dataset_statistics.json"),
        "required_files": ("G0_3B_base.pt", "G0_3B_base_dataset_statistics.json"),
    },
    "dm05": {
        "repo_id": "Dexmal/DM05",
        "label": "OpenDM DM0.5",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "opendm_dm05",
        "allow_patterns": ("*",),
        "required_files": ("model.safetensors", "norm_stats.json", "config.json"),
    },
    "xiaomi-robotics-1": {
        "repo_id": "XiaomiRobotics/Xiaomi-Robotics-1-5B",
        "label": "Xiaomi-Robotics-1 5B",
        "embodiment_id": "emb_robotwin_aloha_agilex",
        "embodiment_env_id": "ee_robotwin_aloha_joint14_v1",
        "framework": "xiaomi_xr1",
        "allow_patterns": ("model_states.pt", "README.md"),
        "required_files": ("model_states.pt",),
    },
}


def source_by_repo(repo_id: str):
    for source_id, source in MODEL_SOURCES.items():
        if source["repo_id"] == repo_id:
            return source_id, source
    raise KeyError(repo_id)
