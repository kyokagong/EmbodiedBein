from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,expected', [
    ('probe_rpo_ufo_dynamics_finalize_v2', []),
    ('probe_rpo_ufo_mjlab_v12', [0]),
])
async def test_cpu_rsi_finalizer_does_not_reserve_legacy_default_gpu(operation, expected):
    from eb.scheduler.loop import Scheduler
    from eb.domain.job import JobSpec, JobKind
    spec = JobSpec(kind=JobKind.ASSET_VALIDATE, embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':operation},
        resources={'gpus':0, 'memory_gb':16})
    job = SimpleNamespace(id='job_cpu', kind=spec.kind.value, embodiment_env_id=spec.embodiment_env_id,
        spec=spec.model_dump(mode='json'))
    session = SimpleNamespace(scalars=AsyncMock(side_effect=[
        SimpleNamespace(all=lambda:[]), SimpleNamespace(all=lambda:[job])]))
    orch = SimpleNamespace(admit_and_start=AsyncMock())
    scheduler = Scheduler(orch)
    scheduler.reap_running = AsyncMock(); scheduler.advance_pipelines = AsyncMock()
    scheduler.ebd = SimpleNamespace(doctor=AsyncMock(return_value={
        'memory_available_mb':80000,
        'gpus':[{'index':0, 'memory_free_mb':24080, 'run_id':None}]}))
    await scheduler.tick(session)
    orch.admit_and_start.assert_awaited_once_with(session, job, expected)
