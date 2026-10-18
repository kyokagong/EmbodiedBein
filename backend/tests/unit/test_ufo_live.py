from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def live_plan(backend="mujoco"):
    from eb.ufo_live import environment_spec, live_suite_spec, suite_id
    from eb.ufo_native import TASKS, plan_hash

    plan = {
        "operation": "ufo_native_live",
        "name": "frozen",
        "model_id": "mv_export",
        "source_model_id": "mv_native",
        "input_path": "/data/kk/embodiedbein/runs/run_source/outputs/payload",
        "source_artifact_sha256": "sha256:" + "a" * 64,
        "native_source_path": "/data/kk/embodiedbein/runs/run_import/outputs/model",
        "native_source_sha256": "sha256:" + "b" * 64,
        "image_digest": "sha256:" + "c" * 64,
        "backend": backend,
        "environment_id": environment_spec(backend)["id"],
        "environment_hash": environment_spec(backend)["content_hash"],
        "suite_id": suite_id(backend),
        "suite_spec": live_suite_spec(backend),
        "tasks": [task["id"] for task in TASKS],
        "initial_task": "move-ego-0-0",
        "duration_seconds": 300,
        "seed": 71000,
    }
    plan["hash"] = plan_hash(plan)
    return plan


def test_live_plan_rejects_changed_bounds_tasks_environment_and_hash():
    from eb.ufo_live import validate_live_plan
    from eb.ufo_native import plan_hash

    plan = live_plan()
    assert validate_live_plan(plan) == plan
    for key, value in [
        ("duration_seconds", 601),
        ("duration_seconds", True),
        ("seed", -1),
        ("initial_task", "shell"),
        ("tasks", ["move-ego-0-0"]),
        ("backend", "isaac"),
        ("environment_hash", "sha256:" + "d" * 64),
        ("environment_id", "ep_other"),
        ("image_digest", "arbitrary:latest"),
    ]:
        broken = {**plan, key: value}
        broken["hash"] = plan_hash(broken)
        with pytest.raises(ValueError):
            validate_live_plan(broken)
    broken = deepcopy(plan)
    broken["seed"] += 1
    with pytest.raises(ValueError):
        validate_live_plan(broken)


@pytest.mark.parametrize("backend,gpus", [("mujoco", 1), ("mjlab", 1)])
def test_live_job_reuses_native_adapter_and_freezes_separate_control_mount(
    tmp_path, monkeypatch, backend, gpus,
):
    from eb.adapters.ufo_native import UfoNativeAdapter
    from eb.api.routes_ufo_native import spec_for
    from eb.ufo_native import validate_job

    plan = live_plan(backend)
    spec = spec_for(plan)
    assert spec.kind == "serve.policy"
    assert spec.resources.gpus == gpus
    assert validate_job(spec) == plan
    monkeypatch.setattr("eb.adapters.ufo_native.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    rendered = UfoNativeAdapter().render(spec, "run_0123456789abcdef0123")
    assert (tmp_path / "runs/run_0123456789abcdef0123/live-control").is_dir()
    assert (str(tmp_path / "runs/run_0123456789abcdef0123/live-control"), "/control", "ro") in rendered.mounts
    assert (plan["native_source_path"], "/native-source", "ro") in rendered.mounts
    for patch in [
        {"resources": {"gpus": 1 - gpus}},
        {"policy": {"timeout_h": 1}},
        {"inputs": {"base_model": "/other"}},
    ]:
        bad = type(spec).model_validate({**spec.model_dump(), **patch})
        with pytest.raises(ValueError):
            validate_job(bad)


@pytest.mark.asyncio
async def test_only_session_owner_human_may_control_and_foreign_run_is_hidden():
    from fastapi import HTTPException

    from eb.api.deps import Principal
    from eb.api.routes_ufo_live import owned_run

    run = SimpleNamespace(id="run_0123456789abcdef0123", job_id="job_live", status="running")
    job = SimpleNamespace(
        created_by="user:kk",
        spec={"params": {"operation": "ufo_native_live", "native_plan": live_plan()}},
        status="running",
    )
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda cls, ident: run if ident == run.id else job))
    assert (await owned_run(session, Principal("kk", "user"), run.id))[0] is run
    for principal in [Principal("kk", "agent"), Principal("other", "user")]:
        with pytest.raises(HTTPException) as error:
            await owned_run(session, principal, run.id)
        assert error.value.status_code in (403, 404)
    run.status = "finished"
    with pytest.raises(HTTPException) as error:
        await owned_run(session, Principal("kk", "user"), run.id, mutation=True)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_live_model_selection_requires_export_and_environment_identity(monkeypatch):
    from eb.api.routes_ufo_live import LiveIn, build_plan

    native = AsyncMock(side_effect=ValueError("此操作要求导出后的推理版本"))
    monkeypatch.setattr("eb.api.routes_ufo_live.native_model", native)
    with pytest.raises(ValueError, match="导出"):
        await build_plan(SimpleNamespace(), LiveIn(model_id="mv_source", environment_id="ep_other"))
    assert native.await_args.kwargs["exported"] is True


@pytest.mark.asyncio
async def test_live_preview_freezes_parent_environment_and_rejects_mismatches(monkeypatch):
    from eb.api.routes_ufo_live import LiveIn, build_plan
    from eb.ufo_live import environment_spec, live_suite_spec
    from eb.ufo_native import ENV_ID

    expected = environment_spec("mujoco")
    environment = SimpleNamespace(**expected, status="active")
    suite = SimpleNamespace(
        id=expected["suite_id"], frozen=True, embodiment_env_id=ENV_ID, spec=live_suite_spec("mujoco")
    )
    artifact = SimpleNamespace(
        sha256="sha256:" + "a" * 64, storage="/data/kk/embodiedbein/runs/run_source/outputs/payload"
    )
    parent = SimpleNamespace(id="mv_native")
    native_artifact = SimpleNamespace(
        sha256="sha256:" + "b" * 64, storage="/data/kk/embodiedbein/runs/run_import/outputs/model"
    )
    model = SimpleNamespace(id="mv_export", name="frozen", card={"source_model_id": parent.id})
    payload = {"source_model_id": parent.id, "source_artifact_sha256": native_artifact.sha256}
    native = AsyncMock(
        side_effect=lambda session, mid, exported: (
            (model, artifact, payload) if exported else (parent, native_artifact, {})
        )
    )
    monkeypatch.setattr("eb.api.routes_ufo_live.native_model", native)
    monkeypatch.setattr("eb.api.routes_ufo_live.approved_image", AsyncMock(return_value="sha256:" + "c" * 64))
    session = SimpleNamespace(
        get=AsyncMock(side_effect=lambda cls, ident: environment if ident == environment.id else suite)
    )
    body = LiveIn(model_id=model.id, environment_id=environment.id, duration_seconds=30)
    plan = await build_plan(session, body)
    assert plan["native_source_sha256"] == native_artifact.sha256
    assert plan["environment_hash"] == expected["content_hash"]
    assert plan["duration_seconds"] == 30
    with pytest.raises(ValueError, match="环境"):
        await build_plan(session, body.model_copy(update={"environment_id": "ep_foreign"}))
    environment.content_hash = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="哈希|环境"):
        await build_plan(session, body)
    environment.content_hash = expected["content_hash"]
    payload["source_model_id"] = "mv_foreign"
    with pytest.raises(ValueError, match="父模型"):
        await build_plan(session, body)


@pytest.mark.asyncio
async def test_generic_native_submission_cannot_forge_live_model_or_profile(monkeypatch):
    from eb.api.routes_ufo_native import spec_for
    from eb.orchestrator.ufo_native import validate_submission
    from eb.ufo_native import plan_hash

    plan = live_plan()
    monkeypatch.setattr("eb.api.routes_ufo_live.build_plan", AsyncMock(return_value=plan))
    await validate_submission(None, spec_for(plan))
    changed = {**plan, "model_id": "mv_foreign"}
    changed["hash"] = plan_hash(changed)
    with pytest.raises(ValueError, match="registered|frozen"):
        await validate_submission(None, spec_for(changed))


def test_live_command_schema_is_strict_and_task_limited():
    from pydantic import ValidationError

    from eb.api.routes_ufo_live import ControlIn

    assert (
        ControlIn(id="safe_1", action="task", task="move-ego-0-0.3").model_dump(exclude_none=True)["task"]
        == "move-ego-0-0.3"
    )
    for value in [
        {"id": "../x", "action": "pause"},
        {"id": "x", "action": "shell"},
        {"id": "x", "action": "task", "task": "other"},
        {"id": "x", "action": "pause", "task": "move-ego-0-0"},
        {"id": "x", "action": "pause", "extra": 1},
    ]:
        with pytest.raises(ValidationError):
            ControlIn.model_validate(value)


@pytest.mark.asyncio
async def test_control_records_accepted_command_in_audit_log(monkeypatch):
    from eb.api.deps import Principal
    from eb.api.routes_ufo_live import ControlIn, control

    rid = "run_0123456789abcdef0123"
    records = []
    session = SimpleNamespace(add=records.append, commit=AsyncMock())
    monkeypatch.setattr(
        "eb.api.routes_ufo_live.owned_run",
        AsyncMock(
            return_value=(
                SimpleNamespace(id=rid),
                SimpleNamespace(id="job_live"),
                live_plan(),
            )
        ),
    )
    receipt = {"id": "pause1", "action": "pause", "received_at": 100.0}
    client = SimpleNamespace(live_control=AsyncMock(return_value=receipt))
    monkeypatch.setattr("eb.api.routes_ufo_live.EbdClient", lambda: client)
    assert (
        await control(rid, ControlIn(id="pause1", action="pause"), session, Principal("kk", "user"))
        == receipt
    )
    assert len(records) == 1
    assert records[0].principal == "user:kk"
    assert records[0].subject == rid
    assert records[0].after == {"job_id": "job_live", "run_id": rid, "id": "pause1", "action": "pause"}
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_proxy_streams_bytes_without_parsing_pose_json(monkeypatch):
    import httpx

    from eb.api.routes_ufo_native import stream_output

    class SnapshotStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"partial":'
            yield b"raw-bytes}"

    def source(request):
        assert request.url.path == "/v1/runs/run_0123456789abcdef0123/live/state"
        return httpx.Response(200, headers={"cache-control": "private, no-store"}, stream=SnapshotStream())

    monkeypatch.setattr(
        "eb.api.routes_ufo_native.httpx.AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(source)
    )
    request = SimpleNamespace(method="GET", headers={})
    result = await stream_output(request, "run_0123456789abcdef0123", "", live_kind="state")
    body = b"".join([chunk async for chunk in result.body_iterator])
    assert body == b'{"partial":raw-bytes}'
    assert result.headers["cache-control"] == "private, no-store"
