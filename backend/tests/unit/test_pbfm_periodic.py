from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eb import pbfm_periodic


@pytest.mark.asyncio
@pytest.mark.parametrize('job_status,run_status,ebd_status', [
    ('running','running','running'), ('cancelled','running','running'),
    ('cancelled','aborted','running'), ('succeeded','finished','exited'),
])
async def test_periodic_registration_requires_cancelled_stopped_source(
    monkeypatch, job_status, run_status, ebd_status,
):
    job=NS(id='job_source',status=job_status,spec={})
    run=NS(id='run_source',job_id=job.id,status=run_status)
    session=NS(get=AsyncMock(return_value=job),scalar=AsyncMock(return_value=run))
    client=NS(get_run=AsyncMock(return_value={'status':ebd_status}))
    monkeypatch.setattr(pbfm_periodic,'EbdClient',lambda:client)
    with pytest.raises(ValueError,match='cancelled|exited'):
        await pbfm_periodic.periodic_context(session,job.id,59968)


@pytest.mark.asyncio
async def test_periodic_model_card_cannot_select_arbitrary_native_path(monkeypatch):
    binding={'schema_version':'eb/pbfm-periodic-model/v1','job_id':'job_source','run_id':'run_source',
             'step':59968,'manifest_sha256':'a'*64,'parent_plan_hash':'sha256:'+'b'*64}
    checkpoint={'native_path':'/managed/runs/run_source/outputs/recovery/generations/u59968/native',
                'counts':{'optimizer_steps':59968},'manifest':{'files':[]}}
    job=NS(id='job_source')
    run=NS(id='run_source')
    monkeypatch.setattr(pbfm_periodic,'periodic_context',AsyncMock(return_value=(job,run,{},checkpoint,binding)))
    model=NS(card={'periodic_checkpoint':binding,'native_path':'/unmanaged/other','counts':checkpoint['counts']})
    with pytest.raises(ValueError,match='identity'):
        await pbfm_periodic.periodic_model_source(NS(),model,job,run)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [
    None, 'missing_shard', 'receipt', 'path', 'rank', 'plan', 'count', 'parameters',
])
async def test_complete_periodic_source_is_bound_to_native_state(tmp_path, monkeypatch, failure):
    import json

    from test_pbfm_longrun_runtime import periodic_fixture, set_counts
    from test_pbfm_training_validation import validation_spec

    from eb.pbfm_training import segment_endpoint
    from eb.pbfm_training_checkpoint import canonical_hash
    from eb.runners.pbfm_longrun import publish_periodic_checkpoint

    staging, plan, ranks = periodic_fixture(tmp_path)
    for rank in ranks:
        rank['parameter_sha256'] = 'sha256:' + 'e' * 64
    endpoint = segment_endpoint(59968, 4, start_updates=plan['start_updates'],
        start_local_time=plan['start_local_time'], num_envs=2048)
    set_counts(staging, ranks, 59968, endpoint // 4)
    root = tmp_path / 'runs/run_long'
    root.mkdir(parents=True)
    recovery = root / 'outputs/recovery'
    index = publish_periodic_checkpoint(staging, recovery, plan, ranks)
    plan['hash'] = canonical_hash(plan)
    (root / 'pbfm-train-plan.json').write_text(json.dumps(plan))
    frozen = {k: v for k, v in plan.items() if k not in {'hash', 'run_id'}}
    spec = validation_spec(tmp_path)
    spec.params['probe_mode'] = 'training_long_exploration'
    job = NS(id='job_long', status='cancelled', spec=spec.model_dump(mode='json'))
    run = NS(id='run_long', job_id=job.id, status='aborted')
    session = NS(get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=run))
    monkeypatch.setattr(pbfm_periodic, 'get_settings', lambda: NS(data_root=tmp_path))
    monkeypatch.setattr(pbfm_periodic, 'EbdClient', lambda: NS(
        get_run=AsyncMock(return_value={'status': 'exited'})))
    monkeypatch.setattr('eb.pbfm_runtime_probe._training_view',
                        lambda _: NS(params={'native_train_plan': frozen}))
    monkeypatch.setattr('eb.pbfm_training.validate_train_job', lambda _: frozen)
    native = recovery / index['latest']['path']
    if failure == 'missing_shard':
        (native / 'checkpoint/buffers/train_rank_2/buffer.hdf5').unlink()
    elif failure == 'receipt':
        (native.parent / 'receipt.json').write_text('{}')
    elif failure == 'path':
        index['generations'][0]['path'] = '../../other/native'
        (recovery / 'index.json').write_text(json.dumps(index))
    elif failure == 'plan':
        plan['identity']['seed'] = 17
        (root / 'pbfm-train-plan.json').write_text(json.dumps(plan))
    elif failure == 'count':
        index['generations'][0]['counts']['local_time'] += 2048
        (recovery / 'index.json').write_text(json.dumps(index))
    elif failure in {'rank', 'parameters'}:
        manifest_path = native / 'checkpoint-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if failure == 'rank':
            manifest['rank_results'][2]['replay_loader_verified'] = False
        else:
            manifest['rank_results'][2]['parameter_sha256'] = 'sha256:' + 'f' * 64
        manifest_path.write_text(json.dumps(manifest))
        digest = pbfm_periodic.sha256(manifest_path)
        index['latest']['manifest_sha256'] = digest
        (recovery / 'index.json').write_text(json.dumps(index))
        receipt = json.loads((native.parent / 'receipt.json').read_text())
        receipt['manifest_sha256'] = digest
        (native.parent / 'receipt.json').write_text(json.dumps(receipt))
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            await pbfm_periodic.periodic_context(session, job.id, 59968)
    else:
        _, _, _, checkpoint, binding = await pbfm_periodic.periodic_context(session, job.id, 59968)
        assert checkpoint['counts']['optimizer_steps'] == binding['step'] == 59968
        assert binding['parent_plan_hash'] == plan['hash']


@pytest.mark.asyncio
async def test_periodic_registration_preview_requires_operator_and_fresh_identity(monkeypatch):
    from fastapi import HTTPException

    from eb.api import routes_pbfm as routes

    body = routes.RegisterPeriodicIn(step=59968, preview_hash='stale')
    with pytest.raises(HTTPException) as blocked:
        await routes.preview_periodic('job_source', body, None, NS(actor_type='agent'))
    assert blocked.value.status_code == 403
    monkeypatch.setattr('eb.api.routes_train._lock_native_train_submission', AsyncMock())
    monkeypatch.setattr(pbfm_periodic, 'periodic_context', AsyncMock(return_value=(None, None, None, {}, {})))
    register = AsyncMock()
    monkeypatch.setattr(pbfm_periodic, 'register_periodic', register)
    with pytest.raises(HTTPException) as stale:
        await routes.register_periodic_checkpoint('job_source', body, NS(), NS(actor_type='user'))
    assert stale.value.status_code == 409
    register.assert_not_awaited()


@pytest.mark.asyncio
async def test_periodic_publication_keeps_cancelled_status_actual_progress_and_is_idempotent(
    tmp_path, monkeypatch,
):
    from unittest.mock import Mock

    from test_pbfm_longrun import long_source, request

    from eb.orchestrator import pbfm_training as collector
    from eb.pbfm_runtime_probe import _training_view, build_spec
    from eb.store.models import Artifact, ModelVersion

    source_session, *_ = await long_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), source_session)
    training = _training_view(spec)
    counts = {'optimizer_steps': 59968, 'local_time': 7696384, 'global_time': 30785536}
    checkpoint = {'step': 59968, 'counts': counts, 'full_state_finite_verified': True,
        'manifest': {'files': [{'path': 'checkpoint/model/model.safetensors', 'bytes': 10,
                               'sha256': 'b' * 64}], 'image_digest': spec.image,
                     'source_refs': training.params['native_train_plan']['source_refs']},
        'native_path': str(tmp_path / 'native'), 'path': str(tmp_path / 'native/checkpoint'),
        'bytes': 10, 'mode': 'pbfm_native_state', 'limitations': ['environment reset'],
        'state_coverage': {'model': 'restored'}}
    binding = {'job_id': 'job_source', 'run_id': 'run_source', 'step': 59968}
    run = NS(id='run_source', status='aborted', metrics_summary={'optimizer_steps': 60016})
    job = NS(id='job_source', status='cancelled', spec=spec.model_dump(mode='json'))
    verify = AsyncMock()
    monkeypatch.setattr('eb.orchestrator.ufo_native.verify_index', verify)
    session = NS(scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None),
                 add=Mock(), flush=AsyncMock())
    mid = await collector.collect_pbfm_training(session, job, run, tmp_path,
        diagnostic_spec=training, periodic_checkpoint=checkpoint, periodic_binding=binding)
    added = [call.args[0] for call in session.add.call_args_list]
    artifacts = [row for row in added if isinstance(row, Artifact)]
    model = next(row for row in added if isinstance(row, ModelVersion))
    assert mid == model.id and model.card['counts'] == counts
    assert model.card['periodic_checkpoint'] == binding and model.card['training_completed'] is False
    assert model.card['train_plan']['target_updates'] == 100000
    assert job.status == 'cancelled' and run.status == 'aborted'
    assert run.metrics_summary['optimizer_steps'] == 60016
    assert 'model_version_id' not in run.metrics_summary and 'training_completed' not in run.metrics_summary
    assert run.metrics_summary['periodic_checkpoints'][0]['registered_model_id'] == mid
    assert len(artifacts) == 2 and all(a.pinned for a in artifacts)
    verify.assert_awaited_once()
    session.add.reset_mock()
    session.scalar.side_effect = [artifacts[0], artifacts[1], model]
    again = await collector.collect_pbfm_training(session, job, run, tmp_path,
        diagnostic_spec=training, periodic_checkpoint=checkpoint, periodic_binding=binding)
    assert again == mid and len(run.metrics_summary['periodic_checkpoints']) == 1
    session.add.assert_not_called()
