"""Opt-in deletion regression against a disposable eb_delete_check_* database."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from eb.api.deps import Principal
from eb.orchestrator.deletion import STORAGE_LOCK, delete_object, guard_storage_inputs
from eb.store.models import (
    Artifact,
    AuditLog,
    Base,
    EmbodimentEnvRow,
    EmbodimentRow,
    EvalReport,
    EvalSuite,
    Job,
    LineageEdge,
    Run,
    TrainJob,
)


async def main():
    dsn = os.environ.get("EB_DELETE_TEST_DSN")
    if not dsn:
        raise SystemExit("Set EB_DELETE_TEST_DSN to an empty disposable eb_delete_check_* database")
    url = make_url(dsn)
    if url.drivername != "postgresql+asyncpg" or not url.database.startswith("eb_delete_check_"):
        raise SystemExit("Refusing to use a non-disposable deletion database")
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    principal = Principal("test", "user")
    try:
        async with engine.begin() as conn:
            assert not await conn.scalar(text("SELECT to_regclass('public.jobs')")), "database is not empty"
            await conn.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add(EmbodimentRow(id="emb", display_name="test", source={}))
            await session.flush()
            session.add(EmbodimentEnvRow(id="env", embodiment_id="emb", spec={}, spec_hash="test"))
            session.add(EvalSuite(id="suite", name="test", embodiment_env_id="env", spec={}))
            await session.flush()
            session.add_all(
                [
                    Job(
                        id="job_train",
                        embodiment_env_id="env",
                        kind="train.rl",
                        spec={},
                        spec_hash="test",
                        status="failed",
                        created_by="user:test",
                    ),
                    Job(
                        id="job_eval",
                        embodiment_env_id="env",
                        kind="eval.sim",
                        spec={},
                        spec_hash="test",
                        status="failed",
                        created_by="user:test",
                    ),
                ]
            )
            await session.flush()
            session.add_all(
                [
                    Run(id="run_first", job_id="job_train", status="aborted"),
                    Run(id="run_second", job_id="job_train", status="finished"),
                    Run(id="run_eval", job_id="job_eval", status="aborted"),
                    TrainJob(
                        id="trn_test",
                        name="test",
                        job_id="job_train",
                        status="failed",
                        template_id="test",
                        embodiment_id="emb",
                        run_dir="",
                    ),
                ]
            )
            await session.flush()
            session.add(Artifact(id="art", run_id="run_eval", sha256="test", kind="video", storage="test"))
            session.add(EvalReport(id="ev", suite_id="suite", run_id="run_eval", metrics={}))
            session.add(
                LineageEdge(
                    src_type="run",
                    src_id="run_eval",
                    dst_type="eval_report",
                    dst_id="ev",
                    relation="produced",
                )
            )
            await session.commit()

        clean = AsyncMock(return_value={"removed_bytes": 12, "paths": []})
        with patch("eb.orchestrator.deletion.EbdClient", return_value=SimpleNamespace(delete_storage=clean)):
            async with sessions() as session:
                plan = await delete_object(session, principal, "train", "trn_test", preview=True)
                assert len(plan["run_ids"]) == 2 and not plan["blockers"]
                clean.assert_not_awaited()
            # Actual DB transactions prove a consumer waits for the exclusive reservation.
            async with sessions() as deleting, sessions() as consuming:
                await deleting.execute(text(f"SELECT pg_advisory_xact_lock({STORAGE_LOCK})"))
                pid = await consuming.scalar(text("SELECT pg_backend_pid()"))
                consumer = asyncio.create_task(guard_storage_inputs(consuming, {"run_id": "run_first"}))
                async with engine.connect() as observer:
                    blockers = []
                    for _ in range(100):
                        blockers = await observer.scalar(text("SELECT pg_blocking_pids(:pid)"), {"pid": pid})
                        if blockers:
                            break
                        await asyncio.sleep(0.02)
                    assert blockers and not consumer.done()
                deleting.add(
                    AuditLog(
                        principal=principal.id,
                        action="storage.delete",
                        subject="probe",
                        before={"ids": ["run_first"], "paths": []},
                        after={"state": "pending"},
                    )
                )
                await deleting.commit()
                try:
                    await consumer
                    raise AssertionError("new consumer accepted a pending deletion")
                except ValueError:
                    pass
                await consuming.rollback()

            # I/O failure keeps metadata + a durable tombstone; retry removes children in FK order.
            clean.side_effect = RuntimeError("simulated ebd failure")
            async with sessions() as session:
                try:
                    await delete_object(session, principal, "eval", "job_eval")
                    raise AssertionError("failed cleanup reported success")
                except HTTPException as exc:
                    assert exc.status_code == 503
            async with sessions() as session:
                receipt = await session.scalar(select(AuditLog).where(AuditLog.subject == "eval:job_eval"))
                assert receipt.after["state"] == "pending" and await session.get(Job, "job_eval")
            clean.side_effect = None
            async with sessions() as session:
                result = await delete_object(session, principal, "eval", "job_eval")
                assert result["state"] == "completed"
            async with sessions() as session:
                for table, key in [
                    (Job, "job_eval"),
                    (Run, "run_eval"),
                    (Artifact, "art"),
                    (EvalReport, "ev"),
                ]:
                    assert await session.get(table, key) is None
                assert await session.scalar(select(LineageEdge))
                before = clean.await_count
                result = await delete_object(session, principal, "eval", "job_eval")
                assert result["already_deleted"] and clean.await_count == before
            async with sessions() as session:
                await delete_object(session, principal, "train", "trn_test")
            async with sessions() as session:
                assert await session.get(TrainJob, "trn_test") is None
                assert not (await session.scalars(select(Run))).all()
        print("PASS: PostgreSQL locking, durable retry, FK cleanup, tombstones, lineage and idempotency")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=30))
