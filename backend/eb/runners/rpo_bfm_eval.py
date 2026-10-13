"""BFM command actor bridge; physical cases reuse the existing RPO evaluator."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def configure_action_control(exported):
    try:
        from eb.rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
    except ModuleNotFoundError:
        from rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
    control = action_control_for_execution(exported.get("execution"))
    if exported.get("action_control") != control:
        raise ValueError("Exported action control differs from frozen source recipe")
    auxiliary = auxiliary_profile_for_execution(exported.get("execution"))
    if exported.get("auxiliary_profile") != auxiliary:
        raise ValueError("Exported auxiliary profile differs from frozen source recipe")
    if auxiliary:
        os.environ["EB_UFO_AUXILIARY_PROFILE"] = auxiliary
    else:
        os.environ.pop("EB_UFO_AUXILIARY_PROFILE", None)
    if control:
        os.environ["EB_UFO_LEARNING_PROFILE"] = "physical_limits_slew_v4"
    else:
        os.environ.pop("EB_UFO_LEARNING_PROFILE", None)


def pack_observation(obs):
    arrays=[]
    batch=None
    for key,width in (("states",52),("last_action",23),("history_actor",300),("privileged_state",358)):
        value=obs[key]
        if hasattr(value,"detach"):value=value.detach().cpu().numpy()
        value=np.asarray(value,dtype=np.float32)
        if value.ndim!=2 or value.shape[1]!=width or value.shape[0]<1 or batch is not None and value.shape[0]!=batch:
            raise ValueError("BFM evaluation native observation contract differs")
        batch=value.shape[0]
        arrays.append(value)
    policy=np.concatenate(arrays[:3],axis=-1)
    return {"policy":policy,"critic":np.concatenate((policy,arrays[3]),axis=-1)}


class CommandActorSession:
    def __init__(self,actor,command_graph):
        self.actor,self.command_graph=actor,command_graph
        self.command=None
        self.last_latent=None

    def get_inputs(self):return self.actor.get_inputs()

    def run(self,outputs,inputs):
        commands=np.asarray(self.command(),dtype=np.float32)
        batch=next(iter(inputs.values())).shape[0]
        if (commands.shape!=(batch,3) or not np.isfinite(commands).all()
                or (commands<[-.60001,-.50001,-1.57001]).any()
                or (commands>[1.00001,.50001,1.57001]).any()):
            raise ValueError("BFM evaluation command exceeds frozen range")
        z=self.command_graph.run(None,{"command":commands})[0].astype(np.float32)
        if z.shape!=(batch,256) or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z,axis=-1),16,atol=1e-3):
            raise ValueError("BFM evaluation command latent is invalid")
        self.last_latent=z.copy()
        return self.actor.run(outputs,{**inputs,"z":z})


def sensor_noise_bounds(scales):
    return np.array([scales.joint_pos]*23+[scales.joint_vel]*23+
        [scales.projected_gravity]*3+[scales.ang_vel/4]*3,dtype=np.float32)


def standard_configuration(cfg):
    from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg
    from robolab.tasks.direct.base.scene_cfg import SceneCfg
    official=RPOFlatEnvCfg()
    cfg.scene_context.terrain_generator=official.scene_context.terrain_generator
    cfg.scene_context.terrain_type=official.scene_context.terrain_type
    cfg.scene=SceneCfg(config=cfg.scene_context,physics_dt=cfg.sim.dt,step_dt=cfg.decimation*cfg.sim.dt)
    cfg.scene.left_feet_scanner.debug_vis=cfg.scene.right_feet_scanner.debug_vis=False
    cfg.noise=official.noise
    return cfg


def standard_environment_type():
    import torch
    from rpo_bfm_extension import RPOBFMIsaacEnv

    class EvaluationEnv(RPOBFMIsaacEnv):
        def _get_observations(self):
            obs=super()._get_observations()
            if self.cfg.noise.add_noise:
                if self.num_envs!=1:raise ValueError("Independent BFM evaluation requires one environment per seed")
                key=(int(self.bfm_episode_id[0]),int(self.bfm_step_id[0]))
                if getattr(self,"_eval_sensor_key",None)!=key:
                    bounds=torch.from_numpy(sensor_noise_bounds(self.cfg.noise.noise_scales)).to(self.device)
                    self._eval_sensor_state=obs["states"]+(2*torch.rand_like(obs["states"])-1)*bounds
                    self._eval_sensor_key=key
                # Repeat reads use the same measurement; history must contain what the actor saw.
                obs["states"]=self._eval_sensor_state.clone()
            return obs
    return EvaluationEnv


def main():
    from rpo_bfm_artifact import export,sha256
    from rpo_flat_artifact import evaluate_isaac
    p=argparse.ArgumentParser()
    for key in ("checkpoint","resolved-config","motion-root","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--asset-sha256",required=True)
    for key in ("weights-sha256","motion-sha256","config-sha256"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--plan-hash",required=True)
    p.add_argument("--seeds",required=True)
    p.add_argument("--record-video",action="store_true")
    p.add_argument("--video-quality",choices=("standard","hd"),default="standard")
    p.add_argument("--diagnostic-commands",action="store_true")
    a=p.parse_args()
    native = (a.checkpoint / "native-checkpoint.json").is_file()
    if native:
        from bfm_ufo_artifact import native_checkpoint_path, export_ufo_policy
        weights = native_checkpoint_path(a.checkpoint) / "model/model.safetensors"
    else:
        weights = a.checkpoint / "model_state_dict/full_weights.pt"
    if (sha256(weights)!=a.weights_sha256
            or sha256(a.motion_root/"manifest.json")!=a.motion_sha256
            or sha256(a.resolved_config)!=a.config_sha256):
        raise ValueError("BFM evaluator input changed since frozen preview")
    (export_ufo_policy if native else export)(a.checkpoint,a.resolved_config,a.motion_root,a.asset_sha256,a.output/"exported")
    receipt=evaluate_isaac(a.output/"exported/policy.onnx",a.output,[int(s) for s in a.seeds.split(",")],
        record_video=a.record_video,video_quality=a.video_quality,plan_hash=a.plan_hash,
        recipe="bfm",diagnostic_commands=a.diagnostic_commands)
    print(json.dumps(receipt),flush=True)
    if receipt["cases"]["infra_failed"]:raise SystemExit(2)


if __name__=="__main__":main()
