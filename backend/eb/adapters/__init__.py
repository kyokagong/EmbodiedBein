from __future__ import annotations

from eb.adapters.base import Adapter
from eb.adapters.data_convert import DataConvertAdapter, DataRetargetAdapter
from eb.adapters.data_scan import DataScanAdapter
from eb.adapters.isaaclab_rslrl import IsaacLabRslRlAdapter
from eb.adapters.mujoco import MujocoAdapter, NativeAdapter, ToolsAdapter
from eb.adapters.rlinf import RlinfAdapter
from eb.domain.job import JobKind, JobSpec

_ISAAC = IsaacLabRslRlAdapter()
_RLINF = RlinfAdapter()
_MUJOCO = MujocoAdapter()
_TOOLS = ToolsAdapter()
_NATIVE = NativeAdapter()
_CONVERT = DataConvertAdapter()
_RETARGET = DataRetargetAdapter()
_SCAN = DataScanAdapter()

KIND_ADAPTER: dict[JobKind, Adapter] = {
    JobKind.TRAIN_RL: _ISAAC,
    JobKind.EVAL_SIM: _ISAAC,
    JobKind.EXPORT_POLICY: _ISAAC,
    JobKind.DATA_COLLECT: _ISAAC,
    JobKind.TRAIN_SFT: _RLINF,
    JobKind.TRAIN_RLFT: _RLINF,
    JobKind.SERVE_POLICY: _RLINF,
    JobKind.EVAL_SIM2SIM: _MUJOCO,
    JobKind.DATA_COMPILE: _TOOLS,
    JobKind.DATA_CONVERT: _CONVERT,
    JobKind.DATA_CANONICALIZE: _TOOLS,
    JobKind.DATA_RETARGET: _RETARGET,
    JobKind.REPORT_RENDER: _TOOLS,
    JobKind.ASSET_VALIDATE: _TOOLS,
    JobKind.BUNDLE_BUILD: _TOOLS,
    JobKind.DATA_SCAN: _SCAN,
    JobKind.DATA_SLICE: _NATIVE,
}


def get_adapter(spec: JobSpec) -> Adapter:
    from eb.ufo_native import ENV_ID, OPERATIONS
    if spec.embodiment_env_id == ENV_ID or spec.params.get('operation') in OPERATIONS:
        from eb.adapters.ufo_native import UfoNativeAdapter
        return UfoNativeAdapter()
    if spec.params.get("stack") == "rlinf" or spec.params.get("template_id") in {"collect_berkeley_reach"}:
        return _RLINF
    return KIND_ADAPTER[spec.kind]


def equivalent_command(spec: JobSpec, run_id: str) -> str:
    rendered = get_adapter(spec).render(spec, run_id)
    gpu = ""
    if rendered.gpus:
        gpu = f' --gpus \'"device={",".join(str(i) for i in range(rendered.gpus))}"\''
    devices = " ".join(f"--device {d}" for d in rendered.devices)
    cmd = " ".join(rendered.command)
    return (
        f"docker run --rm -d --name eb-run-{run_id} --label eb.run_id={run_id}"
        f"{gpu} {devices} --shm-size {rendered.shm_gb}g "
        f"-w {rendered.workdir} {rendered.image} {cmd}"
    ).strip()
