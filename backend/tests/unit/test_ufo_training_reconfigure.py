import json
from copy import deepcopy

import pytest

from eb.api.routes_train import ResumeTrainIn, build_resume_spec
from eb.store.models import Job
from eb.ufo_training import segment_endpoint, training_fingerprint
from eb.ufo_training_checkpoint import validate_reconfigured_resume


def test_2048_fresh_collection_and_later_full_resume_have_distinct_counters():
    assert segment_endpoint(10112, 4, start_updates=10000, start_local_time=655360,
                            num_envs=2048, fresh_replay=True) == 2727936
    assert segment_endpoint(15008, 4, start_updates=10112, start_local_time=681984,
                            num_envs=2048) == 5242880
    with pytest.raises(ValueError):
        segment_endpoint(10112, 4, start_updates=10000, start_local_time=655361,
                         num_envs=2048, fresh_replay=True)


def reconfigured_context(monkeypatch):
    from test_ufo_resume_preview import resume_context
    (_, _, source, checkpoint), _ = resume_context(monkeypatch)
    source.params['native_train_plan'].update(full_state_finite_required=True, source_refs={})
    checkpoint['reconfigure_image_digest'] = 'sha256:' + 'd' * 64
    body = ResumeTrainIn(name='2048 continuation', target_updates=224,
                         native_reconfigure={'num_envs_per_rank':2048, 'buffer_size':524288,
                                             'replay_policy':'fresh'})
    return source, checkpoint, body


def test_reconfigure_preview_freezes_fresh_replay_and_does_not_mutate_source(monkeypatch):
    source, checkpoint, body = reconfigured_context(monkeypatch)
    before = source.model_dump(mode='json')
    result = build_resume_spec(source, 'trn_source', checkpoint, body)
    assert source.model_dump(mode='json') == before
    plan = result.params['native_train_plan']
    assert result.params['resume_mode'] == 'ufo_native_state_fresh_replay'
    assert result.image == checkpoint['reconfigure_image_digest']
    assert plan['identity']['num_envs_per_rank'] == 2048
    assert plan['identity']['buffer_size'] == 524288
    assert plan['fingerprint'] != checkpoint['compatibility_fingerprint']
    assert plan['fingerprint'] == training_fingerprint(plan['identity'])
    assert plan['resume']['source_plan'] == before['params']['native_train_plan']
    assert result.policy == source.policy


def test_2048_keeps_original_replay_capacity_when_measured_peak_requires_margin(monkeypatch):
    source, checkpoint, body = reconfigured_context(monkeypatch)
    body.native_reconfigure.buffer_size = 262144
    result = build_resume_spec(source, 'trn_source', checkpoint, body)
    assert result.params['native_train_plan']['identity']['buffer_size'] == 262144
    assert result.params['native_train_plan']['identity']['num_envs_per_rank'] == 2048
    body.native_reconfigure.buffer_size = 131072
    with pytest.raises(ValueError):
        build_resume_spec(source, 'trn_source', checkpoint, body)


def test_capacity_uses_measured_2048_reserve_and_rejects_oversized_replay():
    from test_ufo_training import native_spec

    from eb.capacity.estimator import estimate_job
    spec = native_spec()
    identity = spec.params['native_train_plan']['identity']
    identity.update(num_envs_per_rank=2048, buffer_size=524288)
    large = estimate_job(spec, host_vram_gb=24)
    identity['buffer_size'] = 262144
    smaller = estimate_job(spec, host_vram_gb=24)
    assert large.verdict == 'no_fit'
    assert smaller.verdict != 'no_fit'
    assert smaller.per_gpu_peak_gb < large.per_gpu_peak_gb


@pytest.mark.parametrize('drift', ['seed','data','world_size','source','buffer','undeclared','image'])
def test_reconfiguration_never_loosens_other_resume_identity_checks(monkeypatch, drift):
    source, checkpoint, body = reconfigured_context(monkeypatch)
    result = build_resume_spec(source, 'trn_source', checkpoint, body)
    plan = result.params['native_train_plan']
    if drift in ('seed','data','world_size','buffer'):
        key = {'data':'dataset_id','buffer':'buffer_size'}.get(drift, drift)
        plan['identity'][key] = 'changed'
    elif drift == 'source':
        plan['source_refs'] = {'ufo':'changed'}
    elif drift == 'image':
        plan['resume']['source_plan']['identity']['image_digest'] = 'changed'
    else:
        plan['resume'].pop('replay_policy')
    with pytest.raises(ValueError):
        validate_reconfigured_resume(plan, source.params['native_train_plan'])


def test_runtime_reconfiguration_copies_optimizer_and_counters_without_old_replay(tmp_path, monkeypatch):
    from eb.runners import rlinf_ufo_runtime as runtime
    source_owner, checkpoint, body = reconfigured_context(monkeypatch)
    result = build_resume_spec(source_owner, 'trn_source', checkpoint, body)
    plan = result.params['native_train_plan']
    root = tmp_path / 'source/native'
    state = root / 'checkpoint'
    shard = state / 'buffers/train_rank_0'
    shard.mkdir(parents=True)
    (shard / 'snapshot.json').write_text(json.dumps({
        'id':'source-replay', 'global_time':18432, 'world_size':1,'transition_format':2}))
    (shard / 'buffer.hdf5').write_bytes(b'old 1024 environments')
    status = {'world_size':1, 'local_time':18432, 'global_time':18432,
              'optimizer_steps':112, 'buffer_global_time':18432,'replay_id':'source-replay'}
    (state / 'train_status.json').write_text(json.dumps(status))
    (state / 'optimizers.pth').write_bytes(b'all five Adam states')
    (root / 'checkpoint-manifest.json').write_text(json.dumps({'run_id':'run_source'}))
    plan['resume']['checkpoint_path'] = str(state)
    monkeypatch.setattr(runtime, 'NATIVE_ROOT', tmp_path / 'target/native')
    calls = []
    monkeypatch.setattr('eb.ufo_training_checkpoint.validate_native_checkpoint',
                        lambda *a, **kw: calls.append(kw))
    runtime.prepare_segment(plan, {'num_envs':2048,'num_env_steps':45056,
                                  'checkpoint_every_steps':45056,'buffer_every_steps':45056})
    target = tmp_path / 'target/native/checkpoint'
    assert not (target / 'buffers').exists()
    assert (target / 'optimizers.pth').read_bytes() == (state / 'optimizers.pth').read_bytes()
    assert (target / 'optimizers.pth').stat().st_ino != (state / 'optimizers.pth').stat().st_ino
    assert json.loads((target / 'train_status.json').read_text()) == status
    assert (shard / 'buffer.hdf5').is_file()
    assert calls[0]['plan'] == source_owner.params['native_train_plan']
    assert calls[0]['verify_hashes'] is True
    receipt = json.loads((tmp_path / 'target/native/initialization.json').read_text())
    assert receipt['replay_per_rank'] == 'reinitialized'


@pytest.mark.asyncio
async def test_admission_reconfiguration_still_binds_the_original_source_job(tmp_path, monkeypatch):
    from test_ufo_training_service import resume_fixture

    from eb.ufo_training_service import validate_submission
    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    old = deepcopy(session.rows[Job].spec['params']['native_train_plan'])
    plan = spec.params['native_train_plan']
    plan['identity'] = {**old['identity'], 'num_envs_per_rank':2048, 'buffer_size':524288,
                        'image_digest':'sha256:'+'d'*64}
    plan['image_digest'] = spec.image = 'sha256:'+'d'*64
    plan['fingerprint'] = training_fingerprint(plan['identity'])
    plan['resume'].update(source_plan=old, replay_policy='fresh')
    session.client.image_info.return_value = {'ok':True, 'digest':spec.image}
    from eb.ufo_training import native_arguments
    plan['native_args'] = native_arguments(plan)
    await validate_submission(session, spec)
    plan['identity']['seed'] = 999
    plan['fingerprint'] = training_fingerprint(plan['identity'])
    plan['native_args'] = native_arguments(plan)
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
