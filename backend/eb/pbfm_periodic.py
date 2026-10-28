"""Register complete periodic state from a stopped P-BFM Run without changing it."""
from pathlib import Path

from sqlalchemy import select

from eb.domain.job import JobSpec
from eb.pbfm_io import hash_value, read_json, safe_file, sha256
from eb.pbfm_training_checkpoint import canonical_hash, validate_pbfm_checkpoint
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Job, Run


async def periodic_context(session, job_id: str, step: int):
    if type(step) is not int or not 112 <= step < 100000 or step % 16:
        raise ValueError("P-BFM periodic step must be a native 16-update boundary")
    job = await session.get(Job, job_id)
    run = await session.scalar(select(Run).where(Run.job_id == job_id).order_by(
        Run.attempt.desc(), Run.created_at.desc()).limit(1))
    if not job or job.status != 'cancelled' or not run or run.job_id != job.id or run.status != 'aborted':
        raise ValueError('P-BFM periodic publication requires a cancelled, stopped training Run')
    state = await EbdClient().get_run(run.id)
    if state.get('status') != 'exited':
        raise ValueError('P-BFM periodic source runtime is not exited')
    from eb.pbfm_runtime_probe import _training_view
    from eb.pbfm_training import segment_endpoint, validate_train_job

    source = JobSpec.model_validate(job.spec)
    if source.params.get('probe_mode') not in {'training_long_exploration', 'training_long_recovery'}:
        raise ValueError('P-BFM periodic publication requires a frozen long-training producer')
    training = _training_view(source)
    frozen = validate_train_job(training)
    root = get_settings().data_root / 'runs' / run.id
    plan = read_json(safe_file(root, 'pbfm-train-plan.json'), maximum=4 * 1024**2)
    if (plan.get('run_id') != run.id or plan.get('hash') != canonical_hash(plan)
            or {k: v for k, v in plan.items() if k not in {'hash', 'run_id'}} != frozen
            or not plan.get('start_updates', 0) < step < plan['target_updates']):
        raise ValueError('P-BFM periodic frozen plan/step differs')
    recovery = root / 'outputs/recovery'
    index = read_json(safe_file(recovery, 'index.json'), maximum=4 * 1024**2)
    if (index.get('schema_version') != 'eb/pbfm-periodic-recovery-index/v1'
            or index.get('run_id') != run.id or index.get('parent_plan_hash') != plan['hash']
            or index.get('parent_target_updates') != plan['target_updates']
            or not isinstance(index.get('generations'), list)
            or not 1 <= len(index['generations']) <= 2 or index.get('latest') != index['generations'][0]):
        raise ValueError('P-BFM periodic index identity differs')
    entries = [row for row in index['generations'] if row.get('counts', {}).get('optimizer_steps') == step]
    if len(entries) != 1:
        raise ValueError('P-BFM requested periodic checkpoint is not completely indexed')
    entry = entries[0]
    expected_global = segment_endpoint(step, plan['world_size'], start_updates=plan.get('start_updates', 0),
        start_local_time=plan.get('start_local_time', 0), num_envs=plan['identity']['num_envs_per_rank'])
    counts = {'optimizer_steps': step, 'local_time': expected_global // plan['world_size'],
              'global_time': expected_global}
    if entry.get('path') != f'generations/u{step}/native' or entry.get('counts') != counts:
        raise ValueError('P-BFM periodic path or actual counters differ')
    manifest_path = safe_file(recovery, entry['path'] + '/checkpoint-manifest.json')
    native = manifest_path.parent
    manifest = read_json(manifest_path, maximum=4 * 1024**2)
    receipt = read_json(safe_file(recovery, f'generations/u{step}/receipt.json'))
    if (sha256(manifest_path) != entry.get('manifest_sha256') or receipt != {
            'schema_version': 'eb/pbfm-periodic-recovery/v1', 'status': 'complete',
            'training_complete': False, 'run_id': run.id, 'parent_plan_hash': plan['hash'], **entry}
            or manifest.get('plan_hash') != plan['hash']):
        raise ValueError('P-BFM periodic atomic receipt/manifest differs')
    checkpoint = validate_pbfm_checkpoint(native, manifest, run_id=run.id, plan=plan, resume=True)
    if checkpoint['counts'] != counts or any(
        set(row.get('optimizer_state_steps', {})) != set(row['optimizer_names'])
        or any(values != [step] for values in row['optimizer_state_steps'].values())
        for row in manifest['rank_results']
    ):
        raise ValueError('P-BFM periodic native optimizer proof differs')
    parameters = {row.get('parameter_sha256') for row in manifest['rank_results']}
    if len(parameters) != 1 or not all(hash_value(value, prefixed=True) for value in parameters):
        raise ValueError('P-BFM periodic rank parameters are inconsistent')
    binding = {'schema_version': 'eb/pbfm-periodic-model/v1', 'job_id': job.id, 'run_id': run.id,
               'step': step, 'manifest_sha256': entry['manifest_sha256'], 'parent_plan_hash': plan['hash']}
    return job, run, training, checkpoint, binding


async def periodic_model_source(session, model, job, run):
    binding = (model.card or {}).get('periodic_checkpoint') or {}
    source, saved_run, _, checkpoint, expected = await periodic_context(session, job.id, binding.get('step'))
    if (binding != expected or source.id != job.id or saved_run.id != run.id
            or model.card.get('native_path') != checkpoint['native_path']
            or model.card.get('counts') != checkpoint['counts']
            or model.card.get('full_state_finite_verified') is not True):
        raise ValueError('P-BFM periodic model identity differs from its immutable source')
    return checkpoint


async def register_periodic(session, job_id: str, step: int):
    from eb.orchestrator.pbfm_training import collect_pbfm_training
    from eb.pbfm_runtime_probe import _training_view

    job, run, training, checkpoint, binding = await periodic_context(session, job_id, step)
    return await collect_pbfm_training(session, job, run, Path(checkpoint['native_path']),
        diagnostic_spec=_training_view(JobSpec.model_validate(job.spec)),
        periodic_checkpoint=checkpoint, periodic_binding=binding)
