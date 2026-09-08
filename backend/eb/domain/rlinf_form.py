from __future__ import annotations

import binascii
from dataclasses import dataclass, field

RAY_BASE = 6390
RAY_SPAN = 50

FORCED = [
    "runner.logger.log_path=/workspace/output",
    "runner.checkpoint.save_dir=/workspace/output/checkpoints",
    "hydra.run.dir=/workspace/output/hydra",
]


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


TEMPLATES: tuple[RlinfTemplate, ...] = (
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


def get_template(template_id: str) -> RlinfTemplate:
    for t in TEMPLATES:
        if t.id == template_id:
            return t
    raise KeyError(template_id)


def _gpu_csv(gpus: int) -> str:
    n = max(1, int(gpus))
    return ",".join(str(i) for i in range(n))


def component_placement(template_id: str, gpus: int) -> str:
    tpl = get_template(template_id)
    csv = _gpu_csv(gpus)
    if tpl.kind == "sft":
        return f"cluster.component_placement.actor_rollout={csv}"
    return f"cluster.component_placement.actor,env,rollout={csv}"


def ray_port_for(job_id: str) -> int:
    crc = binascii.crc32(job_id.encode()) & 0xFFFFFFFF
    return RAY_BASE + (crc % RAY_SPAN)


def forced_overrides(job_name: str) -> list[str]:
    return [
        *FORCED,
        f"runner.logger.experiment_name={job_name}",
    ]


def validate_hydra_overrides(lines: list[str]) -> list[str]:
    errors: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            errors.append(f"缺少 = ：{line}")
            continue
        key, _val = line.split("=", 1)
        if not key.strip():
            errors.append("覆盖键为空")
    return errors


def build_overrides(template_id: str, gpus: int, job_name: str, extra: list[str]) -> list[str]:
    errors = validate_hydra_overrides(extra)
    if errors:
        raise ValueError("; ".join(errors))
    out = [component_placement(template_id, gpus), *forced_overrides(job_name)]
    for line in extra:
        if line.strip():
            out.append(line.strip())
    return out
