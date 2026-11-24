"""Opt-in CPU Bundle Job -> real ebd task container -> PG/CAS integration."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.mark.asyncio
async def test_bundle_job_runs_checked_worker_and_registers_unsigned(monkeypatch):
    database = os.environ.get("EB_TEST_DB_URL", "")
    digest = os.environ.get("EB_TEST_TOOLS_IMAGE_DIGEST", "")
    if not database or not digest:
        pytest.skip("requires isolated PG, ebd and the freshly built tools image digest")
    assert database.rsplit("/",1)[-1] == "eb_data_plane_test", "Never use a platform database"
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient
    from eb.store.models import Base, Artifact, Bundle, EmbodimentRow, EmbodimentEnvRow, ModelFamily, ModelVersion
    from eb.domain.contract import load_env
    from eb.domain.job import JobSpec
    from eb.orchestrator.service import Orchestrator
    get_settings.cache_clear()
    settings = get_settings()
    assert "data-plane" in str(settings.data_root), "Use only the isolated test roots"
    engine = create_async_engine(database)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        image=await EbdClient().image_info("eb/tools:latest")
        assert image.get("ok") is True and image.get("digest")==digest
        # Small static ONNX matrix model; scientific validation itself runs in the Job image.
        import onnx
        from onnx import helper, TensorProto
        source = settings.data_root / "inputs"
        source.mkdir(parents=True,exist_ok=True)
        policy = source / "policy.onnx"
        model = helper.make_model(helper.make_graph([helper.make_node("MatMul",["obs","weight"],["action"])],"identity",
            [helper.make_tensor_value_info("obs",TensorProto.FLOAT,[1,45])],
            [helper.make_tensor_value_info("action",TensorProto.FLOAT,[1,12])],
            initializer=[helper.make_tensor("weight",TensorProto.FLOAT,[45,12],[0.0]*540)]),opset_imports=[helper.make_opsetid("",17)])
        model.ir_version=8
        onnx.save(model, policy)
        from eb.onnx_io import sha256_file
        env = load_env(Path(__file__).resolve().parents[3] / "contracts/ee_bhl_biped_12dof_v1.yaml")
        # Keep the published source object unchanged in this isolated database.
        env = env.model_copy(deep=True)
        env.id="ee_data_plane_test_v1"
        async with sessions() as session:
            session.add(EmbodimentRow(id=env.embodiment_id,display_name="isolated test",source={}))
            await session.flush()
            session.add(EmbodimentEnvRow(id=env.id,embodiment_id=env.embodiment_id,spec=env.model_dump(mode="json"),
                                       spec_hash="test",status="active"))
            session.add(ModelFamily(id="mf_data_plane",name="isolated",family="mlp"))
            session.add(Artifact(id="art_data_plane",sha256=sha256_file(policy),kind="onnx",storage=str(policy),size_bytes=policy.stat().st_size))
            await session.flush()
            session.add(ModelVersion(id="mv_data_plane",family_id="mf_data_plane",name="identity",embodiment_env_id=env.id,artifact_id="art_data_plane"))
            await session.commit()
            spec=JobSpec(kind="bundle.build",embodiment_env_id=env.id,inputs={"base_model":"mv_data_plane"})
            orchestrator=Orchestrator()
            job=await orchestrator.submit(session,spec,"user:test", "data-plane-bundle-test")
            assert job.kind=="bundle.build" and job.status=="queued"
            run=await orchestrator.admit_and_start(session,job,[])
            assert run.status=="running", job.error
            for _ in range(240):
                state=await EbdClient().get_run(run.id)
                if state.get("status")=="exited": break
                await asyncio.sleep(.25)
            else: raise AssertionError("isolated bundle worker did not exit")
            assert state.get("exit_code")==0, (settings.data_root / "runs" / run.id / "console.log").read_text()
            await orchestrator.collect_and_finish(session,job,run)
            assert job.status=="succeeded",job.error
            bundle=await session.scalar(select(Bundle).where(Bundle.model_version_id=="mv_data_plane"))
            assert bundle and not bundle.signed and bundle.manifest["can_sign"] is False
            package=Path(bundle.manifest["cas"])
            assert package.read_bytes()[:4]==bytes.fromhex("28b52ffd")
            import io, tarfile, pyzstd, yaml
            with tarfile.open(fileobj=io.BytesIO(pyzstd.decompress(package.read_bytes()))) as archive:
                assert set(archive.getnames())=={"manifest.yaml","policy.onnx","deploy.yaml","policy_meta.yaml","gate_report.json"}
                archived=yaml.safe_load(archive.extractfile("manifest.yaml").read())
                assert archived["safety"]=={"action_clip":env.action.scale*1.5,"max_cmd_vel":[1.0,.5,.8],
                                           "watchdog_ms":100,"fallback":"freeze_or_damp"}
                assert archived["g3"]["passed"] is True and archived["g4"]["passed"] is True
                assert archived["numerical_check"]["passed"] is True
            original=bundle.id
            from eb.bundle.jobs import collect_bundle
            await collect_bundle(session,job,run)
            await session.commit()
            assert run.metrics_summary["bundle_id"]==original
    finally:
        await engine.dispose()
        get_settings.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure",["rpc","g3"])
async def test_real_pg_failed_claim_discards_earlier_models(monkeypatch,failure):
    database=os.environ.get("EB_TEST_DB_URL","")
    if not database:
        pytest.skip("requires the isolated data-plane test database")
    assert database.rsplit("/",1)[-1]=="eb_data_plane_test"
    import httpx
    from eb.orchestrator.service import Orchestrator
    from eb.store.models import Artifact, Job, ModelVersion, Run
    engine=create_async_engine(database)
    try:
        sessions=async_sessionmaker(engine,expire_on_commit=False)
        async with sessions() as session:
            job=Job(id="job_atomic_"+failure,embodiment_env_id="ee_data_plane_test_v1",kind="export.policy",
                    spec={"params":{}},spec_hash="test",status="running",created_by="user:test")
            session.add(job); await session.flush()
            run=Run(id="run_atomic_"+failure,job_id=job.id,status="running")
            session.add(run); await session.commit()
            async def claim(session,job,run):
                artifact=Artifact(id="art_atomic_"+failure,run_id=run.id,kind="onnx",sha256="a"*64,
                                  size_bytes=4,storage="/isolated-unreferenced-cas")
                session.add(artifact); await session.flush()
                session.add(ModelVersion(id="mv_atomic_"+failure,family_id="mf_data_plane",name="first model",
                                         artifact_id=artifact.id,embodiment_env_id=job.embodiment_env_id))
                await session.flush()
                if failure=="rpc": raise httpx.ConnectError("second file failed")
                job.status="failed"; job.error="G3 failed"; run.status="aborted"; run.abort_reason="g3_failed"
            monkeypatch.setattr("eb.orchestrator.collect.claim_outputs",claim)
            await Orchestrator().collect_and_finish(session,job,run)
            assert await session.get(Artifact,"art_atomic_"+failure) is None
            assert await session.get(ModelVersion,"mv_atomic_"+failure) is None
            await session.refresh(job); await session.refresh(run)
            assert job.status==("infra_failed" if failure=="rpc" else "failed") and run.status=="aborted"
    finally:
        await engine.dispose()
