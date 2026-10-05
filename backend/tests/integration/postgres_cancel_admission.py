"""Standalone opt-in regression; use an empty disposable eb_rowlock_check database."""

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.engine import make_url
from eb.store.models import Base, EmbodimentRow, EmbodimentEnvRow, Job, Run
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.adapters.base import RenderedJob
from eb.api.routes import cancel_job


async def main():
    dsn = os.environ.get('EB_ROWLOCK_TEST_DSN')
    if not dsn:
        print('SKIP: set EB_ROWLOCK_TEST_DSN to an empty disposable PostgreSQL database')
        return
    url = make_url(dsn)
    if url.drivername != 'postgresql+asyncpg' or url.database != 'eb_rowlock_check':
        raise SystemExit('Refusing: use postgresql+asyncpg and disposable database eb_rowlock_check')
    # Never consult application settings or automatically select the production DSN.
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    spec = JobSpec(kind='data.convert', embodiment_env_id='env')
    async with sessions() as seed:
        seed.add(EmbodimentRow(id='emb', display_name='test', source={}))
        await seed.flush()
        seed.add(EmbodimentEnvRow(id='env', embodiment_id='emb', spec={}, spec_hash='test'))
        await seed.flush()
        seed.add(Job(id='job_lock', embodiment_env_id='env', kind='data.convert',
                     spec=spec.model_dump(mode='json'), spec_hash='test', status='queued', created_by='test'))
        await seed.commit()
    entered, release = asyncio.Event(), asyncio.Event()
    async def submit(_):
        entered.set()
        await release.wait()
        return {'status': 'started', 'container_id': 'fake_docker'}
    cancel = AsyncMock()
    try:
        async with sessions() as admission, sessions() as cancellation:
            stale = await cancellation.get(Job, 'job_lock')
            assert stale.status == 'queued'
            pid = await cancellation.scalar(text('select pg_backend_pid()'))
            job = await admission.get(Job, 'job_lock')
            with tempfile.TemporaryDirectory() as directory, \
                 patch('eb.orchestrator.service.prepare_run_dirs', return_value=Path(directory)), \
                 patch('eb.orchestrator.service.get_adapter', return_value=SimpleNamespace(render=lambda *_: RenderedJob('test', ['test'], {}, [], mounts=[]))), \
                 patch('eb.api.routes.EbdClient', return_value=SimpleNamespace(cancel_run=cancel)), \
                 patch('eb.api.routes._job_out', side_effect=lambda j: {'status': j.status}):
                task_a = asyncio.create_task(Orchestrator(SimpleNamespace(submit_run=submit)).admit_and_start(admission, job, []))
                await asyncio.wait_for(entered.wait(), 5)
                task_b = asyncio.create_task(cancel_job('job_lock', cancellation, None))
                # PostgreSQL itself proves B is blocked, rather than relying on a sleep.
                blockers = []
                async with engine.connect() as observer:
                    for _ in range(100):
                        blockers = await observer.scalar(text('select pg_blocking_pids(:pid)'), {'pid': pid})
                        if blockers:
                            break
                        await asyncio.sleep(.02)
                assert blockers and not task_b.done(), 'cancellation did not wait on the admission row lock'
                cancel.assert_not_awaited()
                release.set()
                run = await asyncio.wait_for(task_a, 5)
                result = await asyncio.wait_for(task_b, 5)
                cancel.assert_awaited_once_with(run.id)
                assert result['status'] == 'cancelled'
                assert stale.status == 'cancelled', 'populate_existing did not refresh identity-map instance'
        async with sessions() as check:
            job = await check.get(Job, 'job_lock')
            run = await check.scalar(select(Run).where(Run.job_id == job.id))
            assert job.status == 'cancelled' and run.status == 'aborted'
            assert run.abort_reason == 'cancelled' and run.container_id == 'fake_docker'
        print('PASS: PostgreSQL pg_blocking_pids confirmed B waits for A; stale queued Job refreshed; newly committed Run cancelled and persisted')
    finally:
        release.set()
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(main(), timeout=30))
