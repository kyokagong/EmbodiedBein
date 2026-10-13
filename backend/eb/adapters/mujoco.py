from __future__ import annotations

from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec
from eb.domain.environment_profile import rpo_eval_arguments
from eb.settings import get_settings


class MujocoAdapter(Adapter):
    kind = JobKind.EVAL_SIM2SIM
    name = "mujoco"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        if spec.embodiment_env_id in {"ee_rpo_v2_wbc_23dof_v1", "ee_rpo_v2_amp_locomotion_23dof_v1"}:
            settings = get_settings()
            amp = spec.embodiment_env_id == "ee_rpo_v2_amp_locomotion_23dof_v1"
            if spec.params.get("rpo_checkpoint_eval"):
                checkpoint = Path(spec.inputs.base_model or "")
                if checkpoint.is_dir():
                    checkpoint /= "model_state_dict/full_weights.pt"
                assets = settings.assets_root / "roboparty/rpo-v2"
                mjcf = assets / "rpo_description/mjcf/rpo.xml"
                if not checkpoint.is_file() or not mjcf.is_file():
                    raise ValueError("RPO 评估缺少受管 checkpoint 或 MJCF")
                seeds = spec.params.get("seeds")
                if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 12
                        or any(type(seed) is not int or not 0 <= seed <= 1_000_000 for seed in seeds)
                        or len(seeds) != len(set(seeds))):
                    raise ValueError("RPO 评估 seeds 必须为 1–12 个不重复的非负整数")
                plan = spec.params.get("rpo_eval_plan") or {}
                if plan.get("implicit_seed_defaults") is True and plan.get("seed_override") is False:
                    expected = list(range(3 if plan.get("profile") == "smoke" else 12))
                    if seeds != expected:
                        raise ValueError("RPO 评估计划的默认 seed 与规模不一致")
                seed_args = (["--profile", plan["profile"]]
                             if (plan.get("implicit_seed_defaults") is True
                                 and plan.get("seed_override") is False
                                 and plan.get("profile") in ("smoke", "standard"))
                             else ["--seeds", ",".join(map(str, seeds))])
                return RenderedJob(
                    image=_image(spec, "eb/rlinf:rpo-v2-wbc"),
                    command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                             "/workspace/eb_runner/rpo_wbc_artifact.py", "eval-checkpoint-mujoco",
                             *(["--recipe", "amp"] if amp else []),
                             "--checkpoint", "/workspace/model/full_weights.pt",
                             "--mjcf", "/workspace/rpo/mjcf/rpo.xml",
                             "--output", "/workspace/output", *seed_args,
                             *(["--criterion", plan["criterion"]] if plan.get("criterion") in ("stability", "stand_walk") else []),
                             *rpo_eval_arguments(plan),
                             *(["--record-video"] if spec.params.get("record_video") else [])],
                    env={"MUJOCO_GL":"egl", "PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                    mounts=[(str(checkpoint), "/workspace/model/full_weights.pt", "ro"),
                            (str(assets / "rpo_description"), "/workspace/rpo", "ro"),
                            (str(settings.data_root / "runs" / run_id / "output"), "/workspace/output", "rw")],
                    gpus=1, shm_gb=spec.resources.shm_gb, workdir="/workspace",
                    outputs=["output"], metrics_glob="",
                )
            policy = Path(spec.inputs.base_model or "")
            if policy.is_dir():
                policy = policy / "policy.onnx"
            assets = settings.assets_root / "roboparty/rpo-v2"
            mjcf = assets / "rpo_description/mjcf/rpo.xml"
            if not policy.is_file() or not mjcf.is_file():
                raise ValueError("RPO MuJoCo 评估必须绑定实际 policy.onnx 和受管 MJCF")
            seeds = spec.params.get("seeds", list(range(12)))
            if not seeds or any(type(seed) is not int for seed in seeds):
                raise ValueError("RPO MuJoCo seeds 必须是非空整数列表")
            return RenderedJob(
                image=_image(spec, "eb/mujoco:latest"),
                command=["python", "/workspace/eb_runner/rpo_wbc_artifact.py", "eval-mujoco",
                         "--policy", "/workspace/model/policy.onnx", "--mjcf", "/workspace/rpo/mjcf/rpo.xml",
                         "--output", "/workspace/output", "--seeds", ",".join(map(str, seeds))],
                env={"MUJOCO_GL":"egl", "PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                mounts=[(str(policy), "/workspace/model/policy.onnx", "ro"),
                        (str(assets / "rpo_description"), "/workspace/rpo", "ro"),
                        (str(settings.data_root / "runs" / run_id / "output"), "/workspace/output", "rw")],
                gpus=0, workdir="/workspace", outputs=["output"], metrics_glob="",
            )
        return RenderedJob(
            image=_image(spec, "eb/mujoco:latest"),
            command=[
                "python",
                "scripts/play_mujoco.py",
                "--headless",
                "--export-traj",
                "--seeds",
                ",".join(str(s) for s in spec.params.get("seeds", [0])),
            ],
            env={},
            writable_paths=["outputs"],
            gpus=0,
            workdir="/workspace",
        )


class ToolsAdapter(Adapter):
    name = "tools"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        import json
        from eb.settings import get_settings
        from eb.domain.training_recipe import UFO_ISAAC_IMAGE_DIGEST

        if spec.params.get("operation") in {"probe_rpo_ufo_tracking_contact","probe_rpo_ufo_compliance_v11","probe_rpo_ufo_mjlab_v12","probe_rpo_ufo_mjlab_viability_v13","probe_rpo_ufo_learning_diagnostics_v1","probe_rpo_ufo_learning_diagnostics_v2","probe_rpo_ufo_learning_value_v3","probe_rpo_ufo_learning_diagnostics_v4","probe_rpo_ufo_dynamics_diagnostics_v1","probe_rpo_ufo_dynamics_finalize_v2","probe_rpo_ufo_contact_factor_v1","probe_rpo_ufo_integration_factor_v1","probe_rpo_ufo_action_replay_v1","probe_rpo_ufo_solver_cone_v1","probe_rpo_ufo_control_damping_v1","probe_rpo_ufo_friction_impedance_v1","probe_rpo_ufo_ground_stiffness_v1","probe_rpo_ufo_contact_margin_v1"}:
            import hashlib
            from eb.domain.training_recipe import UFO_TRACKING_CONTACT_IMAGE_DIGEST,UFO_TRACKING_CONTACT_RECIPE,recipe_resources
            operation=spec.params['operation']
            finalize=operation=='probe_rpo_ufo_dynamics_finalize_v2'
            contact_factor=operation=='probe_rpo_ufo_contact_factor_v1'
            margin_factor=operation=='probe_rpo_ufo_contact_margin_v1'
            ground_factor=operation=='probe_rpo_ufo_ground_stiffness_v1' or margin_factor
            friction_factor=operation=='probe_rpo_ufo_friction_impedance_v1' or ground_factor
            solver_factor=operation=='probe_rpo_ufo_solver_cone_v1' or friction_factor
            control_factor=operation=='probe_rpo_ufo_control_damping_v1'
            action_replay=operation=='probe_rpo_ufo_action_replay_v1' or solver_factor or control_factor
            integration_factor=operation=='probe_rpo_ufo_integration_factor_v1' or action_replay
            dynamics=operation=='probe_rpo_ufo_dynamics_diagnostics_v1' or finalize or contact_factor or integration_factor
            expected_params={'operation':operation}
            if contact_factor:
                candidate=spec.params.get('candidate')
                if candidate not in ('B0','C1'):
                    raise ValueError('Contact factor requires frozen B0/C1')
                expected_params['candidate']=candidate
            expected_resources=recipe_resources(UFO_TRACKING_CONTACT_RECIPE)
            prior=None
            if integration_factor:
                import re
                candidate=spec.params.get('candidate')
                source_run=spec.params.get('trace_source_run_id','')
                source_sha=spec.params.get('trace_source_receipt_sha256','')
                if (candidate not in ('I0','I1') or not isinstance(source_run,str)
                        or not re.fullmatch(r'run_[0-9a-f]{20}',source_run) or not isinstance(source_sha,str)
                        or not re.fullmatch(r'[0-9a-f]{64}',source_sha)):
                    raise ValueError('Integration factor requires frozen I0/I1 and a trace source')
                expected_params.update(candidate=candidate,trace_source_run_id=source_run,
                    trace_source_receipt_sha256=source_sha)
                if action_replay:
                    stage=spec.params.get('stage')
                    if stage not in ('capture','replay') or (stage=='capture' and candidate!='I0'):
                        raise ValueError('Action stage requires frozen capture I0 or replay I0/I1')
                    expected_params['stage']=stage
                    if solver_factor and (candidate!='I1' or stage!='replay'):
                        raise ValueError('Solver cone requires frozen I1 replay')
                    if control_factor and (candidate!='I1' or stage!='replay'):
                        raise ValueError('Control damping requires frozen I1 replay')
                prior=get_settings().data_root/'runs'/source_run/'outputs'
                pointer=prior/'dynamics-diagnostics.json'
                if (prior.is_symlink() or pointer.is_symlink() or not pointer.is_file()
                        or pointer.stat().st_size>1024**2 or hashlib.sha256(pointer.read_bytes()).hexdigest()!=source_sha):
                    raise ValueError('Integration trace requires an immutable source receipt')
                header=json.loads(pointer.read_text())
                if (header.get('contact_factor')!='B0' or header.get('baseline_step')!=600
                        or header.get('optimizer_updates')!=0
                        or header.get('integration_factor','I0') not in ('I0','I1')
                        or header.get('dataset_manifest_sha256')!='dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288'):
                    raise ValueError('Integration trace source differs from the frozen baseline')
                if action_replay and stage=='replay' and (header.get('action_program_stage')!='capture'
                        or header.get('integration_factor')!='I0'):
                    raise ValueError('Replay requires the captured I0 action source')
            if finalize:
                expected_params.update(source_run_id='run_f985809e24ce44e3bfba',
                    source_receipt_sha256='65f594b9efe1795e85b2d772f7b752e58a273de8629fdb425a25bc4c72648f7d')
                expected_resources={'gpus':0,'gpu_mode':'exclusive','cpus':8,'memory_gb':16,'shm_gb':1,'devices':[]}
                prior=get_settings().data_root/'runs'/expected_params['source_run_id']/'outputs'
                pointer=prior/'dynamics-diagnostics.json'
                if (prior.is_symlink() or pointer.is_symlink() or not pointer.is_file()
                        or pointer.stat().st_size>1024**2 or hashlib.sha256(pointer.read_bytes()).hexdigest()!=expected_params['source_receipt_sha256']):
                    raise ValueError('RSI finalizer requires the frozen retained physics receipt')
            value_only=operation=='probe_rpo_ufo_learning_value_v3'
            learning=operation in {'probe_rpo_ufo_learning_diagnostics_v1','probe_rpo_ufo_learning_diagnostics_v2','probe_rpo_ufo_learning_diagnostics_v4'} or value_only or dynamics
            compliance=operation=='probe_rpo_ufo_compliance_v11'
            native=operation in {'probe_rpo_ufo_mjlab_v12','probe_rpo_ufo_mjlab_viability_v13'} or learning
            if native:
                from eb.domain.training_recipe import UFO_MJLAB_IMAGE_DIGEST,UFO_MJLAB_VIABILITY_IMAGE_DIGEST
                image=UFO_MJLAB_VIABILITY_IMAGE_DIGEST if operation.endswith('viability_v13') else UFO_MJLAB_IMAGE_DIGEST
                if learning:
                    from eb.domain.training_recipe import UFO_LEARNING_DIAGNOSTIC_IMAGE_DIGEST
                    image=UFO_LEARNING_DIAGNOSTIC_IMAGE_DIGEST
                    if operation.endswith('_v2'):
                        from eb.domain.training_recipe import UFO_LEARNING_DIAGNOSTIC_V2_IMAGE_DIGEST
                        image=UFO_LEARNING_DIAGNOSTIC_V2_IMAGE_DIGEST
                    if value_only:
                        from eb.domain.training_recipe import UFO_LEARNING_VALUE_V3_IMAGE_DIGEST
                        image=UFO_LEARNING_VALUE_V3_IMAGE_DIGEST
                    if operation.endswith('_v4'):
                        from eb.domain.training_recipe import UFO_LEARNING_DIAGNOSTIC_V4_IMAGE_DIGEST
                        image=UFO_LEARNING_DIAGNOSTIC_V4_IMAGE_DIGEST
                    if dynamics:
                        from eb.domain.training_recipe import UFO_DYNAMICS_DIAGNOSTIC_IMAGE_DIGEST
                        image=UFO_DYNAMICS_DIAGNOSTIC_IMAGE_DIGEST
                    if finalize:
                        from eb.domain.training_recipe import UFO_DYNAMICS_FINALIZER_IMAGE_DIGEST
                        image=UFO_DYNAMICS_FINALIZER_IMAGE_DIGEST
                    if contact_factor:
                        from eb.domain.training_recipe import UFO_CONTACT_FACTOR_IMAGE_DIGEST
                        image=UFO_CONTACT_FACTOR_IMAGE_DIGEST
                    if integration_factor:
                        from eb.domain.training_recipe import UFO_INTEGRATION_FACTOR_IMAGE_DIGEST
                        image=UFO_INTEGRATION_FACTOR_IMAGE_DIGEST
                    if action_replay:
                        from eb.domain.training_recipe import UFO_ACTION_REPLAY_IMAGE_DIGEST
                        image=UFO_ACTION_REPLAY_IMAGE_DIGEST
                    if solver_factor:
                        from eb.domain.training_recipe import UFO_SOLVER_CONE_IMAGE_DIGEST
                        image=UFO_SOLVER_CONE_IMAGE_DIGEST
                    if control_factor:
                        from eb.domain.training_recipe import UFO_CONTROL_DAMPING_IMAGE_DIGEST
                        image=UFO_CONTROL_DAMPING_IMAGE_DIGEST
                    if friction_factor:
                        from eb.domain.training_recipe import UFO_FRICTION_IMPEDANCE_IMAGE_DIGEST
                        image=UFO_FRICTION_IMPEDANCE_IMAGE_DIGEST
                    if ground_factor:
                        from eb.domain.training_recipe import UFO_GROUND_STIFFNESS_IMAGE_DIGEST
                        image=UFO_GROUND_STIFFNESS_IMAGE_DIGEST
                    if margin_factor:
                        from eb.domain.training_recipe import UFO_CONTACT_MARGIN_IMAGE_DIGEST
                        image=UFO_CONTACT_MARGIN_IMAGE_DIGEST
            elif compliance:
                from eb.domain.training_recipe import UFO_COMPLIANCE_IMAGE_DIGEST
                image=UFO_COMPLIANCE_IMAGE_DIGEST
            else:image=UFO_TRACKING_CONTACT_IMAGE_DIGEST
            from eb.paths import validate_allowed_path
            from eb.rpo_bfm import CODEC_V2_HASH,ENV_V2_ID
            from eb.runners.rpo_ufo_motion import validate_robot_state_manifest
            if (spec.kind!=JobKind.ASSET_VALIDATE or spec.embodiment_env_id!=ENV_V2_ID
                    or spec.params!=expected_params
                    or spec.resources.model_dump()!=expected_resources
                    or spec.runtime!={"network":"none"} or len(spec.inputs.datasets)!=1
                    or (spec.inputs.base_model and not learning) or spec.inputs.resume_from or spec.overrides
                    or not 0<spec.policy.timeout_h<=1 or spec.policy.max_retries!=0
                    or spec.image not in (None,image)):
                raise ValueError("Tracking/contact diagnostic requires frozen inputs and bounded budget")
            baseline=get_settings().models_root/'runs/run_bc7df2383d8f498bb08d'
            if learning:
                checkpoint=baseline/'native/global_step_600/resume-manifest.json'
                if (spec.inputs.base_model!=str(baseline) or not 0<spec.policy.timeout_h<=.5
                        or checkpoint.is_symlink() or not checkpoint.is_file()
                        or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=
                           'eee3b299fcbad165ff61394c86026b7e08fdf4335f91e588c9c82de159dfc3c7'):
                    raise ValueError('Learning diagnosis requires frozen v13 step600 and 30-minute budget')
            dataset=spec.inputs.datasets[0];source=validate_allowed_path(dataset['path'])
            validation=dataset.get('validation') or {};manifest=source/'manifest.json'
            if learning and validation.get('manifest_sha256')!='dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288':
                raise ValueError('Learning diagnosis requires the frozen v13 train reference manifest')
            asset_root=get_settings().assets_root/'roboparty/rpo-v2';asset=asset_root/'asset-manifest.json'
            if (manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size>1024**2
                    or validation.get('status')!='validated' or validation.get('scope')!='rpo_ufo_robot_state_v2'
                    or asset.is_symlink() or not asset.is_file()
                    or hashlib.sha256(manifest.read_bytes()).hexdigest()!=validation.get('manifest_sha256')
                    or hashlib.sha256(asset.read_bytes()).hexdigest()!=validation.get('asset_sha256')):
                raise ValueError('Tracking/contact diagnostic data or asset changed')
            validate_robot_state_manifest(json.loads(manifest.read_text()),asset_sha256=validation['asset_sha256'],
                expected_codec_sha256=CODEC_V2_HASH)
            output=get_settings().data_root/'runs'/run_id/'outputs'
            script='rpo_ufo_mjlab_probe.py' if native else 'rpo_ufo_compliance_probe.py' if compliance else 'rpo_ufo_tracking_contact_probe.py'
            if learning:script='rpo_ufo_learning_diagnostics.py'
            if dynamics:script='rpo_ufo_dynamics_diagnostics.py'
            if contact_factor:script='rpo_ufo_contact_factor.py'
            if integration_factor:script='rpo_ufo_integration_factor.py'
            if action_replay:script='rpo_ufo_action_replay.py'
            return RenderedJob(image=image,
                command=['/workspace/isaaclab_extension_template/.venv/bin/python',f'/workspace/eb_runner/{script}',
                         *(['--value-only'] if value_only else ['--finalize-rsi'] if finalize else [])],
                env={'RPO_ASSET_ROOT':'/workspace/rpo-assets','EB_UFO_MOTION_ROOT':'/workspace/motion',
                     **({'EB_UFO_PHYSICS_PROFILE':'native_mjcf_viability_v13' if operation.endswith('viability_v13') or learning else 'native_mjcf_v12'} if native else {}),
                     **({'EB_UFO_REWARD_WEIGHTS':json.dumps({'penalty_torques':0.,'penalty_action_rate':-.1,
                           'limits_dof_pos':-10.,'limits_torque':0.,'penalty_undesired_contact':-1.,
                           'penalty_feet_ori':-.4,'penalty_ankle_roll':-4.,'penalty_slippage':-2.,
                           'limits_dof_vel':-25.})} if learning else {}),
                     'EB_UFO_ASSET_SHA256':validation['asset_sha256'],'PYTHONDONTWRITEBYTECODE':'1',
                     **({'EB_UFO_PRIOR_RUN_ID':expected_params['source_run_id']} if finalize else {}),
                     **({'EB_UFO_CONTACT_FACTOR':candidate} if contact_factor else {}),
                     **({'EB_UFO_INTEGRATION_FACTOR':candidate,'EB_UFO_TRACE_SOURCE_RUN_ID':source_run,
                         'EB_UFO_TRACE_SOURCE_SHA256':source_sha} if integration_factor else {}),
                     **({'EB_UFO_ACTION_STAGE':stage} if action_replay else {}),
                     **({'EB_UFO_SOLVER_FACTOR':'S1'} if solver_factor else {}),
                     **({'EB_UFO_CONTROL_FACTOR':'D1'} if control_factor else {}),
                     **({'EB_UFO_FRICTION_FACTOR':'S2'} if friction_factor else {}),
                     **({'EB_UFO_GROUND_STIFFNESS_FACTOR':'C2'} if ground_factor else {}),
                     **({'EB_UFO_CONTACT_MARGIN_FACTOR':'A1'} if margin_factor else {}),
                     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','PXR_WORK_THREAD_LIMIT':'16','PYTHONUNBUFFERED':'1'},
                mounts=[(str(source),'/workspace/motion','ro'),(str(asset_root),'/workspace/rpo-assets','ro'),
                        *([(str(baseline),'/workspace/baseline','ro')] if learning else []),
                        *([(str(prior),'/workspace/prior-evidence','ro')] if finalize else []),
                        *([(str(prior),'/workspace/trace-source','ro')] if integration_factor else []),
                        (str(asset_root/'rpo_description'),'/workspace/rpo-assets/robolab/data/robots/roboparty/rpo','ro'),
                        (str(output),'/outputs','rw')],writable_paths=[],gpus=expected_resources['gpus'],shm_gb=spec.resources.shm_gb,workdir='/workspace')

        if spec.params.get("operation") == "probe_rpo_ufo_contact_v8":
            import hashlib
            from eb.domain.training_recipe import UFO_CONTACT_V8_IMAGE_DIGEST
            from eb.paths import validate_allowed_path
            from eb.rpo_bfm import CODEC_V2_HASH, ENV_V2_ID
            from eb.runners.rpo_ufo_motion import validate_robot_state_manifest
            if (spec.kind != JobKind.ASSET_VALIDATE or spec.embodiment_env_id != ENV_V2_ID
                or spec.params != {"operation": "probe_rpo_ufo_contact_v8"} or spec.resources.gpus != 1
                or spec.runtime != {"network": "none"} or len(spec.inputs.datasets) != 1
                or spec.inputs.base_model or spec.inputs.resume_from or spec.overrides
                or not 0 < spec.policy.timeout_h <= .25 or spec.policy.max_retries != 0
                or spec.image not in (None, UFO_CONTACT_V8_IMAGE_DIGEST)):
                raise ValueError("Contact v8 diagnostic requires frozen bounded one-GPU train inputs")
            dataset=spec.inputs.datasets[0]; source=validate_allowed_path(dataset["path"])
            validation=dataset.get("validation") or {}; manifest=source/"manifest.json"
            asset_root=get_settings().assets_root/"roboparty/rpo-v2"; asset=asset_root/"asset-manifest.json"
            if (manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size>1024**2
                or validation.get("status")!="validated" or validation.get("scope")!="rpo_ufo_robot_state_v2"
                or hashlib.sha256(manifest.read_bytes()).hexdigest()!=validation.get("manifest_sha256")
                or hashlib.sha256(asset.read_bytes()).hexdigest()!=validation.get("asset_sha256")):
                raise ValueError("Contact v8 data/asset differs")
            validate_robot_state_manifest(json.loads(manifest.read_text()),asset_sha256=validation['asset_sha256'],expected_codec_sha256=CODEC_V2_HASH)
            output=get_settings().data_root/"runs"/run_id/"outputs"
            return RenderedJob(image=UFO_CONTACT_V8_IMAGE_DIGEST,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python","/workspace/eb_runner/rpo_ufo_contact_probe.py"],
                env={"RPO_ASSET_ROOT":"/workspace/rpo-assets","EB_UFO_MOTION_ROOT":"/workspace/motion",
                    "EB_UFO_ASSET_SHA256":validation['asset_sha256'],"PYTHONDONTWRITEBYTECODE":"1",
                    "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"4","PXR_WORK_THREAD_LIMIT":"16"},
                mounts=[(str(source),"/workspace/motion","ro"),(str(asset_root),"/workspace/rpo-assets","ro"),
                    (str(asset_root/"rpo_description"),"/workspace/rpo-assets/robolab/data/robots/roboparty/rpo","ro"),
                    (str(output),"/outputs","rw")], writable_paths=[], gpus=1, shm_gb=spec.resources.shm_gb, workdir="/workspace")

        if spec.params.get("operation") in {"probe_rpo_ufo_environment", "probe_rpo_ufo_tracking", "probe_rpo_ufo_initial_contact"}:
            tracking = spec.params["operation"] == "probe_rpo_ufo_tracking"
            contact = spec.params["operation"] == "probe_rpo_ufo_initial_contact"
            from eb.domain.training_recipe import UFO_CONTACT_IMAGE_DIGEST, UFO_TRACKING_IMAGE_DIGEST
            approved_image = UFO_CONTACT_IMAGE_DIGEST if contact else UFO_TRACKING_IMAGE_DIGEST if tracking else UFO_ISAAC_IMAGE_DIGEST
            if contact and spec.params != {"operation": "probe_rpo_ufo_initial_contact"}:
                raise ValueError("RSI contact probe protects its fixed paired protocol")
            import hashlib
            from eb.paths import validate_allowed_path
            from eb.runners.rpo_ufo_motion import validate_robot_state_manifest
            if (spec.kind != JobKind.ASSET_VALIDATE or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.resources.gpus != 1 or spec.runtime != {"network": "none"}
                    or len(spec.inputs.datasets) != 1 or bool(spec.inputs.base_model) != tracking or spec.inputs.resume_from
                    or spec.overrides or not 0 < spec.policy.timeout_h <= .25 or spec.policy.max_retries != 0):
                raise ValueError("UFO environment probe requires one frozen DatasetVersion and bounded one-GPU Job")
            dataset = spec.inputs.datasets[0]
            source = validate_allowed_path(dataset["path"])
            asset_root = get_settings().assets_root / "roboparty/rpo-v2"
            asset, manifest = asset_root / "asset-manifest.json", source / "manifest.json"
            validation = dataset.get("validation") or {}
            if (source.is_symlink() or manifest.is_symlink() or not manifest.is_file()
                    or manifest.stat().st_size > 1024**2 or asset.is_symlink() or not asset.is_file()
                    or validation.get("status") != "validated" or validation.get("scope") != "rpo_ufo_robot_state_v1"
                    or hashlib.sha256(manifest.read_bytes()).hexdigest() != validation.get("manifest_sha256")
                    or hashlib.sha256(asset.read_bytes()).hexdigest() != validation.get("asset_sha256")):
                raise ValueError("UFO environment probe frozen data/asset differs")
            validate_robot_state_manifest(json.loads(manifest.read_text()), asset_sha256=validation["asset_sha256"])
            output = get_settings().data_root / "runs" / run_id / "outputs"
            extra_mounts, extra_env = [], ({"EB_UFO_LEARNING_PROFILE": "physical_limits_v2"} if contact else {})
            if tracking:
                import re
                from eb.checkpoints import checkpoint_model
                from eb.runners.bfm_ufo_tracking import SUITE, tracking_references
                checkpoint = validate_allowed_path(spec.inputs.base_model)
                supplied_plan = spec.params.get("tracking_plan")
                if not isinstance(supplied_plan, dict):
                    raise ValueError("Tracking needs a frozen plan")
                source_run_id = supplied_plan.get("source_run_id", "")
                if not isinstance(source_run_id, str) or not re.fullmatch(r"run_[a-z0-9]+", source_run_id):
                    raise ValueError("Tracking needs a registered source Run")
                metadata = checkpoint_model(get_settings().data_root / "runs" / source_run_id, source_run_id, verify_hashes=False)
                if checkpoint.resolve() != Path(metadata["path"]).resolve() or metadata["manifest"].get("format") != "ufo_agent_state_v1":
                    raise ValueError("Tracking source differs from complete registered native model")
                from eb.domain.training_recipe import UFO_SLEW_RECIPE, UFO_STABLE_RECIPE, UFO_VELOCITY_RECIPE, UFO_SCALE_RECIPE, recipe_eval_image_digest
                if metadata["manifest"]["execution"]["recipe_id"] in {UFO_STABLE_RECIPE, UFO_SLEW_RECIPE, UFO_VELOCITY_RECIPE, UFO_SCALE_RECIPE}:
                    approved_image = recipe_eval_image_digest(metadata["manifest"]["execution"]["recipe_id"])
                weights = next(item for item in metadata["manifest"]["files"] if item["path"] == metadata["manifest"]["weights_path"])
                plan = {"suite": SUITE, "checkpoint_step": metadata["step"], "source_run_id": source_run_id, "image_digest": approved_image,
                    "motion_manifest_sha256": validation["manifest_sha256"], "asset_sha256": validation["asset_sha256"],
                    "weights_sha256": weights["sha256"], "clip_ids": [clip["id"] for clip in tracking_references(json.loads(manifest.read_text())["clips"])]}
                from eb.rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
                control = action_control_for_execution(metadata["manifest"]["execution"])
                if control:
                    plan["action_control"] = control
                auxiliary = auxiliary_profile_for_execution(metadata["manifest"]["execution"])
                if auxiliary:
                    plan["auxiliary_profile"] = auxiliary
                plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if spec.params.get("tracking_plan") != plan:
                    raise ValueError("UFO held-out tracking plan differs from frozen checkpoint/references")
                extra_mounts = [(str(checkpoint), "/workspace/checkpoint", "ro")]
                extra_env = {"EB_UFO_TRACKING_PLAN": json.dumps(plan), "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4", "PXR_WORK_THREAD_LIMIT": "16"}
            if spec.image not in (None, approved_image, "eb/rlinf:rpo-bfm-ufo-agent-v1"):
                raise ValueError("UFO environment probe requires its compatible measured image")
            return RenderedJob(image=approved_image,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                    "/workspace/eb_runner/rpo_ufo_initial_contact.py"] if contact else ["/workspace/isaaclab_extension_template/.venv/bin/python",
                    "/workspace/tests/rpo_ufo_tracking_diagnostic.py"] if tracking else ["/workspace/isaaclab_extension_template/.venv/bin/python",
                    "/workspace/tests/rpo_ufo_environment_smoke.py", "--output", "/outputs/probe.json"],
                env={"RPO_ASSET_ROOT": "/workspace/rpo-assets", "EB_UFO_MOTION_ROOT": "/workspace/motion",
                    "EB_UFO_ASSET_SHA256": validation["asset_sha256"], "PYTHONDONTWRITEBYTECODE": "1", **extra_env},
                mounts=[(str(source), "/workspace/motion", "ro"), (str(asset_root), "/workspace/rpo-assets", "ro"),
                    (str(asset_root / "rpo_description"), "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                    (str(output), "/outputs", "rw"), *extra_mounts], writable_paths=[], gpus=1,
                shm_gb=spec.resources.shm_gb, workdir="/workspace")
        if spec.params.get("operation") == "probe_rpo_bfm_ufo_agent":
            if (spec.kind != JobKind.ASSET_VALIDATE or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.resources.gpus != 1 or spec.runtime != {"network": "none"}
                    or spec.image not in (None, "eb/rlinf:rpo-bfm-ufo-agent-v1")
                    or spec.inputs.datasets or spec.inputs.base_model or spec.overrides
                    or not 0 < spec.policy.timeout_h <= .25 or spec.policy.max_retries != 0):
                raise ValueError("UFO agent probe requires fixed offline one-GPU operation bounded to 15min")
            output = get_settings().data_root / "runs" / run_id / "outputs"
            return RenderedJob(image=UFO_ISAAC_IMAGE_DIGEST,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                    "/workspace/tests/bfm_ufo_agent_smoke.py", "--device", "cuda", "--output", "/outputs/probe.json"],
                env={"PYTHONDONTWRITEBYTECODE": "1"}, mounts=[(str(output), "/outputs", "rw")],
                writable_paths=[], gpus=1, shm_gb=spec.resources.shm_gb, workdir="/workspace")
        if spec.params.get("operation") in {"acquire_g1_lafan_trial", "inspect_g1_lafan_trial", "export_g1_lafan_trial"}:
            import hashlib
            import re
            from eb.settings import get_settings
            from eb.adapters.data_convert import _download_env

            acquire = spec.params["operation"] == "acquire_g1_lafan_trial"
            export = spec.params["operation"] == "export_g1_lafan_trial"
            image = "eb/tools:rpo-g1-export-v2" if export else "eb/tools:rpo-g1-source-v1"
            if (spec.kind != JobKind.ASSET_VALIDATE or spec.resources.gpus != 0
                    or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.runtime.get("network") != ("bridge" if acquire else "none")
                    or spec.params.get("source_use") != "local_noncommercial_no_redistribution"
                    or spec.image not in (None, image)):
                raise ValueError("G1 trial requires fixed CPU resources and local noncommercial source use")
            settings = get_settings()
            output = settings.data_root / "runs" / run_id / "outputs"
            mounts = [(str(output), "/outputs", "rw")]
            command = ["/workspace/isaaclab_extension_template/.venv/bin/python",
                "/workspace/eb_runner/g1_lafan_export.py" if export else "/workspace/eb_runner/g1_lafan_source.py",
                *([] if export else ["--phase", "acquire" if acquire else "inspect"]),
                "--output", "/outputs/g1" if acquire else "/outputs/inspection.json"]
            if export:
                command[-1] = "/outputs/g1-export"
            if not acquire:
                source_run = str(spec.params.get("source_run_id", ""))
                if not re.fullmatch(r"run_[0-9a-f]+", source_run):
                    raise ValueError("G1 trial source Run is invalid")
                source = settings.data_root / "runs" / source_run / "outputs/g1"
                receipt = source / "source-manifest.json"
                if (source.is_symlink() or receipt.is_symlink() or not receipt.is_file()
                        or receipt.stat().st_size > 1024*1024
                        or hashlib.sha256(receipt.read_bytes()).hexdigest() != spec.params.get("source_manifest_sha256")):
                    raise ValueError("G1 trial requires a hash-bound source receipt")
                mounts.append((str(source), "/workspace/source", "ro"))
                command += ["--source", "/workspace/source"]
            return RenderedJob(image=image, command=command,
                env={**(_download_env(settings) if acquire else {}), "PYTHONUNBUFFERED":"1"},
                workdir="/workspace", mounts=mounts, writable_paths=[], gpus=0, shm_gb=1)
        if spec.params.get("operation") in {"retarget_100style_probe", "probe_100style_retarget", "probe_100style_tracking_teacher"}:
            import hashlib
            import json
            import re

            from eb.settings import get_settings

            tracking = spec.params["operation"] == "probe_100style_tracking_teacher"
            physical = spec.params["operation"] != "retarget_100style_probe"
            if (spec.kind != JobKind.ASSET_VALIDATE or spec.resources.gpus != int(physical)
                    or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.runtime.get("network") != "none"
                    or spec.image not in (None, "eb/rlinf:rpo-v2-bfm-zero")):
                raise ValueError("100STYLE RPO trial requires fixed offline retarget/replay resources")
            source_run = str(spec.params.get("source_run_id", ""))
            if not re.fullmatch(r"run_[0-9a-f]+", source_run):
                raise ValueError("100STYLE RPO trial source Run is invalid")
            settings = get_settings()
            source = settings.data_root / "runs" / source_run / "outputs" / ("retargeted" if physical else "100style")
            receipt = source / ("retarget-receipt.json" if physical else "source-manifest.json")
            assets = settings.assets_root / "roboparty/rpo-v2"
            asset_manifest = assets / "asset-manifest.json"
            if (source.is_symlink() or receipt.is_symlink() or not receipt.is_file()
                    or receipt.stat().st_size > 1024*1024
                    or hashlib.sha256(receipt.read_bytes()).hexdigest() != spec.params.get("source_manifest_sha256")
                    or asset_manifest.is_symlink() or not asset_manifest.is_file()
                    or json.loads(asset_manifest.read_text()).get("status") != "verified"):
                raise ValueError("100STYLE RPO trial needs a hash-bound source receipt and verified assets")
            output = settings.data_root / "runs" / run_id / "outputs"
            command = ["/workspace/isaaclab_extension_template/.venv/bin/python",
                "/workspace/eb_runner/rpo_bfm_motion_probe.py" if physical else "/workspace/eb_runner/rpo_100style_retarget.py"]
            command += (["--retarget-dir", "/workspace/source", "--asset-manifest", "/workspace/rpo-assets/asset-manifest.json",
                         "--output", "/outputs/probe.json"] if physical else
                        ["--source", "/workspace/source", "--assets", "/workspace/rpo-assets", "--output", "/outputs/retargeted"])
            teacher_mounts = []
            if tracking:
                teacher_run = str(spec.params.get("teacher_run_id", ""))
                teacher_sha = str(spec.params.get("teacher_weights_sha256", ""))
                if not re.fullmatch(r"run_[0-9a-f]+", teacher_run) or not re.fullmatch(r"[0-9a-f]{64}", teacher_sha):
                    raise ValueError("Tracking teacher source/weight hash is invalid")
                weights = settings.models_root / "runs" / teacher_run / "model_state_dict/full_weights.pt"
                package = weights.parents[1] / "actor-normalization.json"
                if (weights.is_symlink() or not weights.is_file() or package.is_symlink()
                        or hashlib.sha256(weights.read_bytes()).hexdigest() != teacher_sha
                        or json.loads(package.read_text()).get("schema_version") != "eb/rpo-tracking-normalization/v1"):
                    raise ValueError("Tracking teacher checkpoint/normalization changed")
                teacher_mounts = [(str(weights), "/workspace/teacher/full_weights.pt", "ro")]
                command += ["--teacher-weights", "/workspace/teacher/full_weights.pt",
                            "--teacher-sha256", teacher_sha, "--video"]
            return RenderedJob(image="eb/rlinf:rpo-v2-bfm-zero", command=command,
                env={"PYTHONUNBUFFERED":"1", "PYTHONPATH":"/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_tasks:/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf"},
                workdir="/workspace", mounts=[(str(source), "/workspace/source", "ro"),
                    (str(assets), "/workspace/rpo-assets", "ro"),
                    (str(assets/"rpo_description"), "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                    (str(output), "/outputs", "rw"), *teacher_mounts], writable_paths=[], gpus=int(physical),
                shm_gb=spec.resources.shm_gb)
        if spec.params.get("operation") == "acquire_100style_probe":
            from eb.settings import get_settings
            from eb.adapters.data_convert import _download_env

            if (spec.kind != JobKind.ASSET_VALIDATE or spec.resources.gpus != 0
                    or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.runtime.get("network") != "bridge"
                    or spec.image not in (None, "eb/rlinf:rpo-v2-bfm-zero")):
                raise ValueError("100STYLE acquisition requires the fixed CPU source-trial operation")
            settings = get_settings()
            output = settings.data_root / "runs" / run_id / "outputs"
            return RenderedJob(image="eb/rlinf:rpo-v2-bfm-zero",
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_100style_probe.py", "--output", "/outputs/100style"],
                env={**_download_env(settings), "PYTHONUNBUFFERED":"1"}, workdir="/workspace",
                mounts=[(str(output), "/outputs", "rw")], writable_paths=[], gpus=0, shm_gb=1)
        if spec.params.get("operation") in {"probe_rpo_amp_env", "probe_rpo_amp_motion", "probe_rpo_bfm_motion"}:
            import json
            import re
            from eb.settings import get_settings

            bfm = spec.params["operation"] == "probe_rpo_bfm_motion"
            from eb.rpo_bfm import REFERENCE_IMAGE
            image = REFERENCE_IMAGE if bfm else "eb/rlinf:rpo-v2-wbc"
            interface = "ee_rpo_v2_bfm_23dof_v1" if bfm else "ee_rpo_v2_amp_locomotion_23dof_v1"
            if (spec.kind != JobKind.ASSET_VALIDATE
                    or spec.embodiment_env_id != interface
                    or spec.resources.gpus != 1 or spec.runtime.get("network") != "none"
                    or spec.image not in (None, image)):
                raise ValueError("RPO AMP probe requires one isolated Isaac GPU and the pinned image")
            settings = get_settings()
            assets = settings.assets_root / "roboparty/rpo-v2"
            manifest = assets / "asset-manifest.json"
            if (assets.is_symlink() or manifest.is_symlink() or not manifest.is_file()
                    or json.loads(manifest.read_text()).get("status") != "verified"
                    or not (assets / "rpo_description").is_dir()):
                raise ValueError("RPO AMP probe requires verified managed assets")
            output = settings.data_root / "runs" / run_id / "outputs"
            motion = bfm or spec.params.get("operation") == "probe_rpo_amp_motion"
            initial_pose = spec.params.get("initial_pose")
            if initial_pose not in (None, "official_a1_stand_v1"):
                raise ValueError("RPO AMP probe initial pose is unsupported")
            dataset_id = str(spec.params.get("dataset_id", ""))
            dataset = settings.datasets_root / dataset_id / "content" if motion else None
            if motion and (not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)
                           or dataset.is_symlink() or not (dataset / "manifest.json").is_file()):
                raise ValueError("RPO AMP motion probe needs a managed motion DatasetVersion")
            return RenderedJob(
                image=image,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_bfm_motion_probe.py" if bfm else "/workspace/eb_runner/rpo_amp_probe.py", "--output", "/outputs/probe.json",
                         *(["--motion-dir", "/workspace/motion", "--asset-manifest",
                            "/workspace/rpo-assets/asset-manifest.json"] if motion else [])],
                env={"RPO_ASSET_ROOT":"/workspace/rpo-assets", "PYTHONUNBUFFERED":"1",
                     "PYTHONPATH":"/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_tasks:/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf",
                     **({"EB_RPO_AMP_INITIAL_POSE":initial_pose} if initial_pose else {})},
                workdir="/workspace", gpus=1, shm_gb=spec.resources.shm_gb,
                mounts=[(str(assets), "/workspace/rpo-assets", "ro"),
                        (str(assets / "rpo_description"),
                         "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                        *([(str(dataset), "/workspace/motion", "ro")] if motion else []),
                        (str(output), "/outputs", "rw")], writable_paths=[],
            )
        kind = spec.kind.value
        return RenderedJob(
            image=_image(spec, "eb/tools:latest"),
            command=["python", "-m", "eb_tools", kind, "--run-id", run_id],
            env={},
            writable_paths=["outputs"],
            gpus=0,
        )


class NativeAdapter(Adapter):
    name = "native"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        return RenderedJob(
            image="",
            command=["ebd", spec.kind.value, "--run-id", run_id],
            env={},
            writable_paths=[],
            gpus=0,
        )
