import pytest
from eb.adapters import get_adapter
from eb.adapters.rlinf import _eval_plan_env
from eb.domain.job import JobKind, JobSpec
from eb.runners.rlinf_entry import load_eval_plan_env


def test_large_eval_plan_is_compressed_below_exec_environment_limit(monkeypatch):
    plan = {"cases":[{"case_id":f"task/clean/{index:06d}", "task_id":"task",
                       "scenario":"clean", "seed":index} for index in range(5000)]}
    env = _eval_plan_env(plan)

    assert "EB_EVAL_PLAN" not in env
    assert len(env["EB_EVAL_PLAN_ZLIB"]) < 128 * 1024
    monkeypatch.delenv("EB_EVAL_PLAN", raising=False)
    monkeypatch.setenv("EB_EVAL_PLAN_ZLIB", env["EB_EVAL_PLAN_ZLIB"])
    assert load_eval_plan_env() == plan


def make_spec(**kw):
    values = dict(kind=JobKind.TRAIN_SFT, embodiment_env_id='ee_libero_franka_delta_v1', task_id='tsk_libero_spatial', params={'template_id':'libero_pi05_sft','stack':'rlinf'}, inputs={'datasets':[{'id':'dsv_1','path':'/data/libero'}], 'base_model':'/data/pi05'}, resources={'gpus':4,'shm_gb':32})
    values.update(kw)
    return JobSpec(**values)


def test_rpo_ppo_mounts_managed_assets_and_uses_protected_runner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    assets = tmp_path / "assets/roboparty/rpo-v2"
    for relative in ("robolab/robolab/tasks/direct/base/base_env.py",
                     "rpo_description/urdf/rpo.urdf",
                     "rpo_description/mjcf/rpo.xml"):
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned")
    (assets / "asset-manifest.json").write_text('{"status":"verified","hardware_verified":false}')
    settings = SimpleNamespace(
        data_root=tmp_path / "state", models_root=tmp_path / "models",
        rlinf_repo=tmp_path / "RLinf", assets_root=tmp_path / "assets",
    )
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: settings)
    spec = JobSpec(
        kind=JobKind.TRAIN_RL,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        task_id="tsk_rpo_walk_upperbody_tracking",
        params={"template_id": "rpo_v2_wbc_ppo", "stack": "rlinf"},
        resources={"gpus": 1, "shm_gb": 16},
    )

    rendered = get_adapter(spec).render(spec, "run_rpo")

    assert (str(assets), "/workspace/rpo-assets", "ro") in rendered.mounts
    assert (str(assets / "rpo_description"), "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro") in rendered.mounts
    assert rendered.command[1] == "/workspace/eb_runner/rlinf_entry.py"
    assert rendered.env["RPO_ASSET_ROOT"] == "/workspace/rpo-assets"
    assert rendered.env["PYTHONPATH"].startswith("/workspace/isaaclab/source/isaaclab:")
    assert rendered.outputs == ["output"]

    spec.overrides = ["runner.max_epochs=100", "runner.max_steps=100"]
    long_run = get_adapter(spec).render(spec, "run_rpo_long")
    assert "runner.max_epochs=100" in long_run.command
    assert "runner.max_steps=100" in long_run.command

    spec.inputs.datasets = [{"id":"wrong", "path":"/data/wrong"}]
    with pytest.raises(ValueError, match="数据集|大模型"):
        get_adapter(spec).render(spec, "run_rpo")


def test_argv_and_real_input_binding():
    rendered = get_adapter(make_spec()).render(make_spec(), 'run_test')
    assert rendered.command[0] != 'bash'
    assert 'data.train_data_paths=/workspace/data/dsv_1' in rendered.command
    assert 'actor.model.model_path=/workspace/models/base' in rendered.command
    assert any(m[1] == '/workspace/data/dsv_1' and m[2] == 'ro' for m in rendered.mounts)
    assert any(m[0].endswith('/models/runs/run_test') and m[1] == '/workspace/output/model' and m[2] == 'rw' for m in rendered.mounts)
    assert 'EMBODIED_PATH' in rendered.env


def test_no_missing_input_or_runtime_override():
    spec = make_spec()
    spec.inputs.base_model = None
    with pytest.raises(ValueError, match='model|模型'):
        get_adapter(spec).render(spec, 'run_test')
    spec = make_spec(runtime={'env':{'PYTHONPATH':'/evil'}})
    with pytest.raises(ValueError):
        get_adapter(spec).render(spec, 'run_test')


def test_template_cannot_run_under_another_robot_contract():
    spec = make_spec()
    spec.embodiment_env_id = 'ee_bhl_fixedhand_10dof_v1'
    spec.task_id = 'tsk_bhl_reach'
    with pytest.raises(ValueError, match='模板'):
        get_adapter(spec).render(spec, 'run_test')


def test_rlinf_resume_is_read_only_and_uses_protected_runner_path(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from eb.checkpoints import resume_fingerprint

    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / 'models',
                               rlinf_repo=tmp_path / 'rlinf')
    monkeypatch.setattr('eb.adapters.rlinf.get_settings', lambda: settings)
    source = make_spec()
    fingerprint = resume_fingerprint(source)
    checkpoint = tmp_path / 'runs/run_source/output/exp/checkpoints/global_step_80'
    files = []
    for relative in ('actor/dcp_checkpoint/.metadata', 'actor/dcp_checkpoint/__0_0.distcp',
                     'actor/model_state_dict/full_weights.pt', 'data.pt', 'rng.pt'):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        files.append({'path': relative, 'bytes': path.stat().st_size})
    (checkpoint / 'resume-manifest.json').write_text(json.dumps({
        'schema_version':'eb/resume-checkpoint/v1', 'status':'complete', 'run_id':'run_source',
        'step':80, 'mode':'exact', 'compatibility_fingerprint':fingerprint, 'files':files,
        'state_coverage':{'model':'restored','optimizer':'restored','scheduler':'restored',
            'rng_per_rank':'restored','dataloader':'restored','environment':'not_applicable','rollout':'not_applicable'},
    }))
    source.inputs.resume_from = 'run_source'
    source.params.update({'resume_mode':'exact','resume_step':80,'source_train_job_id':'trn_source'})
    source.overrides.append('runner.max_steps=200')

    rendered = get_adapter(source).render(source, 'run_child')

    assert (str(checkpoint.resolve()), '/workspace/resume/global_step_80', 'ro') in rendered.mounts
    assert 'runner.resume_dir=/workspace/resume/global_step_80' in rendered.command
    receipt = json.loads(rendered.env['EB_RESUME_RECEIPT'])
    assert receipt['source_run_id'] == 'run_source' and receipt['step'] == 80
    assert rendered.env['EB_RESUME_FINGERPRINT'] == fingerprint

    import shutil
    selected = checkpoint.parent / 'global_step_60'
    shutil.copytree(checkpoint, selected)
    manifest = json.loads((selected / 'resume-manifest.json').read_text())
    manifest['step'] = 60
    (selected / 'resume-manifest.json').write_text(json.dumps(manifest))
    source.params['resume_step'] = 60
    rendered = get_adapter(source).render(source, 'run_selected')
    assert (str(selected.resolve()), '/workspace/resume/global_step_60', 'ro') in rendered.mounts


def test_rlinf_resume_rejects_missing_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / 'models',
                               rlinf_repo=tmp_path / 'rlinf')
    monkeypatch.setattr('eb.adapters.rlinf.get_settings', lambda: settings)
    spec = make_spec(inputs={'datasets':[{'id':'dsv_1','path':'/data/libero'}],
                             'base_model':'/data/pi05', 'resume_from':'run_missing'})
    with pytest.raises(ValueError, match='source Run|resumable'):
        get_adapter(spec).render(spec, 'run_child')


def test_rlinf_resume_rejects_path_traversal_source():
    spec = make_spec(inputs={'datasets':[{'id':'dsv_1','path':'/data/libero'}],
                             'base_model':'/data/pi05', 'resume_from':'../escape'})
    with pytest.raises(ValueError, match='Run ID'):
        get_adapter(spec).render(spec, 'run_child')


def test_libero_eval_mounts_legacy_checkpoint_read_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eb.adapters.rlinf import RlinfAdapter
    source = tmp_path / 'runs/run_source/output/exp/checkpoints/global_step_5800'
    weight = source / 'actor/model_state_dict/full_weights.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / 'models',
                               rlinf_repo=tmp_path / 'rlinf')
    monkeypatch.setattr('eb.adapters.rlinf.get_settings', lambda: settings)
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_libero_franka_delta_v1',
        task_id='tsk_libero_spatial', params={'template_id':'libero_spatial_pi05_eval', 'stack':'rlinf',
            'eval_checkpoint_run_id':'run_source', 'eval_checkpoint_step':5800,
            'eval_checkpoint_bytes':len(b'weights'), 'eval_source_kind':'sft'},
        inputs={'base_model':str(tmp_path / 'base')}, resources={'gpus':1})
    rendered = RlinfAdapter().render(spec, 'run_eval')
    assert (str(weight.resolve()), '/workspace/checkpoint/full_weights.pt', 'ro') in rendered.mounts
    assert 'runner.ckpt_path=/workspace/checkpoint/full_weights.pt' in rendered.command
    assert 'actor.model.add_value_head=False' in rendered.command
    assert 'actor.model.num_action_chunks=10' in rendered.command
    assert 'actor.model.num_steps=5' in rendered.command
    weight.write_bytes(b'changed-size')
    with pytest.raises(ValueError, match='changed|变更|大小'):
        RlinfAdapter().render(spec, 'run_eval')


def test_robotwin_eval_mounts_frozen_model_source_and_explicit_plan(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from eb.adapters.rlinf import RlinfAdapter

    model = tmp_path / 'model'
    (model / 'physical-intelligence/robotwin').mkdir(parents=True)
    (model / 'physical-intelligence/robotwin/norm_stats.json').write_text('{}')
    (model / 'model_state_dict').mkdir()
    (model / 'model_state_dict/full_weights.pt').write_bytes(b'weights')
    robotwin = tmp_path / 'robotwin'
    (robotwin / 'assets').mkdir(parents=True)
    (robotwin / 'robotwin/envs/vector_env.py').parent.mkdir(parents=True)
    (robotwin / 'robotwin/envs/vector_env.py').write_text('# pinned')
    (robotwin / '.git/refs/heads').mkdir(parents=True)
    (robotwin / '.git/HEAD').write_text('ref: refs/heads/RLinf_support\n')
    (robotwin / '.git/refs/heads/RLinf_support').write_text('abc123\n')
    assets = tmp_path / 'robotwin-assets'
    (assets / 'embodiments/aloha-agilex').mkdir(parents=True)
    (assets / 'objects').mkdir()
    from eb.robotwin_suite import asset_manifest_sha256
    asset_receipt = {'schema_version':'eb/robotwin-assets/v1', 'status':'verified'}
    asset_receipt['sha256'] = asset_manifest_sha256(asset_receipt)
    (assets / 'asset-manifest.json').write_text(json.dumps(asset_receipt))
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / 'models',
                               datasets_root=tmp_path / 'datasets', rlinf_repo=tmp_path / 'rlinf',
                               robotwin_repo=robotwin, robotwin_assets_root=assets)
    (settings.rlinf_repo / '.git/refs/heads').mkdir(parents=True)
    (settings.rlinf_repo / '.git/HEAD').write_text('ref: refs/heads/main\n')
    (settings.rlinf_repo / '.git/refs/heads/main').write_text('def456\n')
    monkeypatch.setattr('eb.adapters.rlinf.get_settings', lambda: settings)
    cases = [{'case_id':f'adjust_bottle/clean/{i:06d}', 'task_id':'adjust_bottle',
              'seed':1000+i, 'scenario':'clean'} for i in range(5)]
    from eb.runners.robotwin_source import PATCH_SHA256, PATCHED_BASE_TASK_SHA256
    plan = {'schema_version':'eb/robotwin-eval-plan/v1', 'suite_id':'suite_robotwin_adjust_bottle_clean_v1',
            'suite_spec':{'robotwin_revision':'abc123', 'rlinf_revision':'def456', 'max_episode_steps':200,
                          'assets_manifest_sha256':asset_receipt['sha256']},
            'profile':'smoke', 'model':{'model_version_id':'mv_pi05', 'runtime':{
                'gpu_count':1, 'policy_worker_count':1, 'env_worker_count':1,
                'envs_per_worker':4, 'parallel_envs':4, 'inference_micro_batch_size':4,
                'max_inflight_batches':1, 'headless':True, 'offscreen_rendering':True,
                'save_video':False, 'reuse_policy_across_tasks':True,
                'rlinf_content_hash':'sha256:' + 'a' * 64,
                'robotwin_evidence_patch_sha256':PATCH_SHA256,
                'patched_base_task_sha256':PATCHED_BASE_TASK_SHA256}},
            'planned':5, 'cases':cases, 'hash':'sha256:' + 'a' * 64}
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_robotwin_aloha_joint14_v1',
        task_id='tsk_robotwin_multitask', params={'template_id':'robotwin_pi05_eval', 'stack':'rlinf',
            'eval_plan':plan, 'suite_id':plan['suite_id'], 'model_version_id':'mv_pi05'},
        inputs={'base_model':str(model)}, resources={'gpus':1, 'shm_gb':16})

    rendered = RlinfAdapter().render(spec, 'run_eval')

    assert 'evaluations/eval_embodied_agent.py' in rendered.command
    assert 'examples/embodiment/train_embodied_agent.py' not in rendered.command
    assert (str(model), '/workspace/models/base', 'ro') in rendered.mounts
    assert (str(robotwin), '/workspace/robotwin', 'ro') in rendered.mounts
    assert (str(assets), '/workspace/robotwin/assets', 'ro') in rendered.mounts
    assert 'env.eval.total_num_envs=4' in rendered.command
    assert 'env.eval.max_episode_steps=200' in rendered.command
    assert 'env.eval.seeds_path=/workspace/output/robotwin-eval-seeds.json' in rendered.command
    assert 'rollout.inference_micro_batch_size=4' in rendered.command
    assert 'rollout.max_inflight_batches=1' in rendered.command
    # RoboTwin appends ``assets/...`` to ASSETS_PATH internally.  The asset
    # bundle is mounted below the source root, so the configured root must be
    # /workspace/robotwin rather than the nested assets directory.
    assert 'env.eval.assets_path=/workspace/robotwin' in rendered.command
    assert 'runner.ckpt_path=/workspace/models/base/model_state_dict/full_weights.pt' in rendered.command
    assert 'cluster.component_placement={env:0,rollout:0}' in rendered.command
    assert 'runner.enable_decoupled_mode=False' in rendered.command
    assert 'rollout.enable_offload=False' in rendered.command
    assert 'env.eval.video_cfg.save_video=False' in rendered.command
    assert rendered.env['NVIDIA_DRIVER_CAPABILITIES'] == 'compute,utility,graphics'
    assert rendered.env['MUJOCO_GL'] == 'egl'
    assert rendered.env['PYOPENGL_PLATFORM'] == 'egl'
    assert rendered.env['ROBOTWIN_PATH'] == '/workspace/robotwin'
    assert rendered.env['ROBOT_PLATFORM'] == 'ALOHA'
    assert json.loads(rendered.env['EB_EVAL_PLAN'])['cases'] == cases

    snapshot = tmp_path / 'snapshots/rlinf/code'
    snapshot.mkdir(parents=True)
    plan['model']['runtime']['rlinf_content_hash'] = 'sha256:' + 'b' * 64
    spec.params['eval_plan'] = plan
    from eb.domain.job import CodeRefs
    spec.code_refs = CodeRefs(stack={'kind':'content_snapshot', 'path':str(snapshot),
                                     'commit':'sha256:' + 'b' * 64})
    rendered = RlinfAdapter().render(spec, 'run_snapshot')
    assert (str(snapshot), '/workspace/source-receipts/rlinf', 'ro') in rendered.mounts
    assert not any(destination == '/workspace/rlinf'
                   for _, destination, _ in rendered.mounts)
    assert rendered.env['EB_RLINF_CONTENT_HASH'] == 'sha256:' + 'b' * 64
    assert rendered.env['EB_RLINF_PATCH_SHA256'].startswith('sha256:')
    spec.code_refs.stack.commit = 'sha256:' + 'c' * 64
    with pytest.raises(ValueError, match='RLInf.*快照'):
        RlinfAdapter().render(spec, 'run_bad_snapshot')
