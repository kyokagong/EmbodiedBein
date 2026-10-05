from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from eb.api.routes_train import LiberoPipelineIn, TrainIn, _build_spec, _memory_gb, list_train
from eb.store.models import Dataset, EmbodimentRow, TrainJob, WeightModel


def test_robotwin_pi05_four_gpu_memory_covers_fsdp_initialization():
    assert _memory_gb("robotwin_pi05_sft", 4) == 72


@pytest.mark.asyncio
async def test_training_list_includes_online_rl_jobs():
    row = TrainJob(id="trn_rpo", name="rpo", status="queued", template_id="rpo_v2_wbc_ppo",
        image_digest="eb/rlinf:rpo-v2-wbc", embodiment_id="emb_roboparty_rpo", dataset_ids=[],
        gpus_allocated=[], ray_port=6390, params={}, hydra_overrides=[], run_dir="", job_id="job_rpo")
    job = SimpleNamespace(kind="train.rl", status="queued", error=None, spec={"inputs": {}})
    session = SimpleNamespace(
        scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [row])),
        get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=None))

    result = await list_train(session, SimpleNamespace(id="user:test"))

    assert [item["id"] for item in result["items"]] == ["trn_rpo"]


@pytest.mark.asyncio
async def test_rpo_assets_are_managed_mounts_not_source_snapshots():
    embodiment = SimpleNamespace(
        root_path="/data/kk/assets/roboparty/rpo-v2",
        source={"kind":"managed_bundle", "path":"/data/kk/assets/roboparty/rpo-v2"},
    )
    session = SimpleNamespace(get=AsyncMock(return_value=embodiment))
    body = TrainIn(name="rpo smoke", model_name="rpo model", template_id="rpo_v2_wbc_ppo",
                   embodiment_id="emb_roboparty_rpo", gpus=2, max_steps=100,
                   total_num_envs=1024)

    spec = await _build_spec(body, session)

    assert spec.code_refs.stack is not None
    assert spec.code_refs.embodiment is None
    assert "runner.max_epochs=100" in spec.overrides


@pytest.mark.asyncio
async def test_training_form_hyperparameters_reach_job_spec():
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(root_path="", source={})))
    body = TrainIn(name="rpo task", model_name="步态模型", template_id="rpo_v2_wbc_ppo",
                   embodiment_id="emb_roboparty_rpo", max_epochs=20, eval_num_envs=32,
                   rollout_epoch=2, save_interval=50, val_check_interval=25,
                   lr="2e-4", value_lr="1e-4", seed=7,
                   extra_overrides=["runner.save_interval=75"])

    spec = await _build_spec(body, session)

    assert spec.params["model_name"] == "步态模型"
    assert "runner.max_epochs=20" in spec.overrides
    assert "env.eval.total_num_envs=32" in spec.overrides
    assert "env.train.rollout_epoch=2" in spec.overrides
    assert "actor.optim.lr=2e-4" in spec.overrides
    assert "actor.optim.value_lr=1e-4" in spec.overrides
    assert "actor.seed=7" in spec.overrides
    assert spec.overrides[-1] == "runner.save_interval=75"


@pytest.mark.asyncio
async def test_training_requires_explicit_model_name():
    body = TrainIn(name="rpo task", template_id="rpo_v2_wbc_ppo",
                   embodiment_id="emb_roboparty_rpo")
    with pytest.raises(ValueError, match="产出模型名称不能为空"):
        await _build_spec(body, SimpleNamespace(get=AsyncMock()))


@pytest.mark.asyncio
async def test_submission_binds_validated_inputs_and_rejects_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv('EB_ALLOWED_PATH_PREFIXES', str(tmp_path))
    from eb.settings import get_settings
    get_settings.cache_clear()
    data = tmp_path / 'dataset'; data.mkdir()
    model = tmp_path / 'model'; model.mkdir()
    rows = {
        (EmbodimentRow, 'emb_libero_franka'): SimpleNamespace(root_path='/data/kk/RLinf', source={'path':'/data/kk/RLinf'}),
        (Dataset, 'ds_test'): SimpleNamespace(id='ds_test', status='validated', dest_path=str(data), source_uri='lerobot/libero',
            embodiment_env_id='ee_libero_franka_delta_v1', meta={}, summary_stats={'features':{'action':{'shape':[7]},'observation.state':{'shape':[8]}},'validation':{'status':'validated','scope':'lerobot_v3_structure','storage_ref':str(data)}}),
        (WeightModel, 'mdl_test'): SimpleNamespace(path=str(model), embodiment_id=None, kind="base", metrics_at_checkpoint={}),
    }
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda table,key:rows.get((table,key))))
    body = TrainIn(name='libero test', model_name='libero model', template_id='libero_pi05_sft', embodiment_id='emb_libero_franka', gpus=4, dataset_ids=['ds_test'], base_model_id='mdl_test')
    try:
        spec = await _build_spec(body, session)
        assert spec.inputs.datasets[0]['path'] == str(data)
        assert spec.inputs.base_model == str(model)
        body.base_model_id = 'missing'
        with pytest.raises(ValueError, match='模型不存在'):
            await _build_spec(body, session)
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_submission_rejects_dataset_from_another_embodiment_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('EB_ALLOWED_PATH_PREFIXES', str(tmp_path))
    from eb.settings import get_settings
    get_settings.cache_clear()
    data = tmp_path / 'dataset'; data.mkdir()
    rows = {
        (EmbodimentRow, 'emb_berkeley_humanoid_lite'): SimpleNamespace(
            root_path='/data/kk/Berkeley-Humanoid-Lite', source={'path':'/data/kk/Berkeley-Humanoid-Lite'}),
        (Dataset, 'ds_wrong'): SimpleNamespace(
            id='ds_wrong', status='validated', dest_path=str(data), source_uri='lerobot/libero',
            embodiment_env_id='ee_libero_franka_delta_v1', meta={},
            summary_stats={'validation': {'status':'validated', 'scope':'lerobot_v3_structure',
                                          'storage_ref':str(data)}}),
    }
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda table, key: rows.get((table, key))))
    body = TrainIn(name='wrong robot data', model_name='reach model', template_id='berkeley_reach_sft',
                   embodiment_id='emb_berkeley_humanoid_lite', dataset_ids=['ds_wrong'])
    try:
        with pytest.raises(ValueError, match='本体环境'):
            await _build_spec(body, session)
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_submission_rejects_eval_labeled_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv('EB_ALLOWED_PATH_PREFIXES', str(tmp_path))
    from eb.settings import get_settings
    get_settings.cache_clear()
    data = tmp_path / 'dataset'; data.mkdir()
    model = tmp_path / 'model'; model.mkdir()
    rows = {
        (EmbodimentRow, 'emb_libero_franka'): SimpleNamespace(root_path='/data/kk/RLinf', source={'path':'/data/kk/RLinf'}),
        (Dataset, 'ds_eval'): SimpleNamespace(id='ds_eval', status='validated', dest_path=str(data),
            source_uri='lerobot/libero', embodiment_env_id='ee_libero_franka_delta_v1',
            meta={'usage':'eval'}, summary_stats={'validation':{'status':'validated',
                'scope':'lerobot_v3_structure','storage_ref':str(data)}}),
        (WeightModel, 'mdl_base'): SimpleNamespace(path=str(model), embodiment_id='emb_libero_franka',
            kind='base', metrics_at_checkpoint={}),
    }
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda table, key: rows.get((table, key))))
    body = TrainIn(name='bad split', model_name='libero model', template_id='libero_pi05_sft', embodiment_id='emb_libero_franka',
                   dataset_ids=['ds_eval'], base_model_id='mdl_base')
    try:
        with pytest.raises(ValueError, match='评估集'):
            await _build_spec(body, session)
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_pipeline_rejects_unadmittable_later_stage_before_saving(monkeypatch):
    from unittest.mock import Mock
    from fastapi import HTTPException
    from eb.api.routes_train import create_libero_pipeline
    from eb.domain.job import JobSpec, JobKind
    spec = JobSpec(kind=JobKind.TRAIN_SFT, name='pipeline', task_id='tsk_libero_spatial',
        embodiment_env_id='ee_libero_franka_delta_v1', resources={'gpus':2},
        params={'template_id':'libero_pi05_sft','stack':'rlinf'})
    monkeypatch.setattr('eb.api.routes_train._build_spec', AsyncMock(return_value=spec))
    monkeypatch.setattr('eb.capacity.estimate_job', lambda job:SimpleNamespace(
        verdict='no_fit' if job.kind==JobKind.TRAIN_RLFT else 'fits'))
    session = SimpleNamespace(add=Mock(),commit=AsyncMock())
    body = LiberoPipelineIn(name='pipeline',template_id='libero_pi05_sft',embodiment_id='emb_libero_franka')
    with pytest.raises(HTTPException) as error:
        await create_libero_pipeline(body,session,SimpleNamespace(id='user:test'))
    assert error.value.status_code == 400
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_stops_at_training_before_independent_evaluation(monkeypatch):
    from unittest.mock import Mock
    from eb.api.routes_train import create_libero_pipeline
    from eb.domain.job import JobSpec, JobKind
    spec = JobSpec(kind=JobKind.TRAIN_SFT, name='pipeline', task_id='tsk_libero_spatial',
        embodiment_env_id='ee_libero_franka_delta_v1', resources={'gpus':4, 'memory_gb':72},
        params={'template_id':'libero_pi05_sft','stack':'rlinf'})
    monkeypatch.setattr('eb.api.routes_train._build_spec', AsyncMock(return_value=spec))
    session = SimpleNamespace(add=Mock(),commit=AsyncMock())
    body = LiberoPipelineIn(name='pipeline',template_id='libero_pi05_sft',embodiment_id='emb_libero_franka',gpus=4)
    result = await create_libero_pipeline(body,session,SimpleNamespace(id='user:test'))
    assert [s['job']['resources']['gpus'] for s in result['spec']['steps']] == [4,4]
    assert [s['job']['resources']['memory_gb'] for s in result['spec']['steps']] == [72,76]
    assert result['spec']['steps'][1]['job']['overrides'] == ['runner.max_epochs=1']


@pytest.mark.asyncio
async def test_pipeline_ppo_epochs_independent_from_sft_and_checked_before_save(monkeypatch):
    from unittest.mock import Mock
    from eb.api.routes_train import create_libero_pipeline
    from eb.domain.job import JobSpec, JobKind
    spec = JobSpec(kind=JobKind.TRAIN_SFT, name='pipeline', task_id='tsk_libero_spatial',
        embodiment_env_id='ee_libero_franka_delta_v1', resources={'gpus':4},
        overrides=['runner.max_steps=2', 'runner.max_epochs=7'],
        params={'template_id':'libero_pi05_sft','stack':'rlinf'})
    monkeypatch.setattr('eb.api.routes_train._build_spec', AsyncMock(return_value=spec))
    checked = []
    def estimate(job):
        checked.append(job)
        return SimpleNamespace(verdict='fits')
    monkeypatch.setattr('eb.capacity.estimate_job', estimate)
    session = SimpleNamespace(add=Mock(), commit=AsyncMock())
    body = LiberoPipelineIn(name='pipeline', template_id='libero_pi05_sft',
                           embodiment_id='emb_libero_franka', gpus=4, max_steps=2,
                           max_epochs=7, ppo_max_epochs=2)
    result = await create_libero_pipeline(body, session, SimpleNamespace(id='test'))
    overrides = [step['job']['overrides'] for step in result['spec']['steps']]
    assert overrides == [['runner.max_steps=2', 'runner.max_epochs=7'], ['runner.max_epochs=2']]
    assert [job.overrides for job in checked] == overrides
    assert spec.overrides == ['runner.max_steps=2', 'runner.max_epochs=7']


@pytest.mark.parametrize('epochs', [0, -1, 10001, 1.5, True, None])
def test_pipeline_rejects_invalid_ppo_epochs(epochs):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LiberoPipelineIn(name='pipeline', template_id='libero_pi05_sft',
                         embodiment_id='emb_libero_franka', ppo_max_epochs=epochs)


@pytest.mark.asyncio
async def test_eval_checkpoint_must_belong_to_terminal_compatible_training(tmp_path, monkeypatch):
    from eb.store.models import Job, Run, TrainJob
    monkeypatch.setenv('EB_ALLOWED_PATH_PREFIXES', str(tmp_path))
    from eb.settings import get_settings
    get_settings.cache_clear()
    model = tmp_path / 'base'; model.mkdir()
    weight = tmp_path / 'runs/run_source/output/exp/checkpoints/global_step_5800/actor/model_state_dict/full_weights.pt'
    weight.parent.mkdir(parents=True); weight.write_bytes(b'weights')
    source = SimpleNamespace(id='trn_source', job_id='job_source', status='running',
        template_id='libero_pi05_sft', embodiment_id='emb_libero_franka', base_model_id='mdl_base')
    rows = {
        (EmbodimentRow, 'emb_libero_franka'): SimpleNamespace(root_path='/data/kk/RLinf', source={'path':'/data/kk/RLinf'}),
        (WeightModel, 'mdl_base'): SimpleNamespace(path=str(model), embodiment_id='emb_libero_franka',
            kind='base', metrics_at_checkpoint={}),
        (Run, 'run_source'): SimpleNamespace(job_id='job_source'),
        (Job, 'job_source'): SimpleNamespace(kind='train.sft', status='failed'),
    }
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda table, key: rows.get((table, key))),
                              scalar=AsyncMock(return_value=source))
    body = TrainIn(name='eval 5800', template_id='libero_spatial_pi05_eval',
        embodiment_id='emb_libero_franka', base_model_id='mdl_base', gpus=1,
        checkpoint_run_id='run_source', checkpoint_step=5800)
    monkeypatch.setattr('eb.api.routes_train.get_settings', lambda: SimpleNamespace(
        data_root=tmp_path, rlinf_repo=tmp_path / 'rlinf'))
    try:
        spec = await _build_spec(body, session)
        assert spec.params['eval_checkpoint_step'] == 5800
        assert spec.params['eval_checkpoint_bytes'] == 7
        assert spec.params['eval_source_kind'] == 'sft'
        body.dataset_ids = ['ds_eval']
        with pytest.raises(ValueError, match='EvalSuite'):
            await _build_spec(body, session)
        body.dataset_ids = []
        source.base_model_id = 'wrong'
        with pytest.raises(ValueError, match='基座'):
            await _build_spec(body, session)
    finally:
        get_settings.cache_clear()
