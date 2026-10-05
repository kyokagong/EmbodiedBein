from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.environment_profile import profile_hash, validate_definition
from eb.domain.rlinf_form import get_template
from eb.environment_catalog import evaluation_definition, suite_executable, suite_task_id, training_definition
from eb.ids import new_id
from eb.store.models import EnvironmentProfile, EvalSuite

router = APIRouter(prefix="/environments")


class EnvironmentEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    settings: dict = Field(default_factory=dict)
    template_id: str | None = None
    suite_id: str | None = None


def _out(row: EnvironmentProfile) -> dict:
    return {
        key: getattr(row, key)
        for key in (
            "id",
            "name",
            "usage",
            "embodiment_env_id",
            "task_id",
            "template_id",
            "suite_id",
            "definition",
            "content_hash",
            "status",
            "revision_of",
        )
    }


def _user(principal) -> None:
    if principal.actor_type != "user":
        raise HTTPException(403, "只有用户可以编辑环境")


@router.get("")
async def list_environments(session: SessionDep, principal: PrincipalDep, usage: str | None = None) -> dict:
    if usage not in (None, "train", "eval"):
        raise HTTPException(400, "usage 必须是 train 或 eval")
    rows = (await session.scalars(select(EnvironmentProfile).order_by(EnvironmentProfile.name))).all()
    return {"items": [_out(row) for row in rows if usage is None or row.usage == usage]}


@router.get("/{env_id}")
async def get_environment(env_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EnvironmentProfile, env_id)
    if not row:
        raise HTTPException(404, "环境不存在")
    return _out(row)


@router.post("/{env_id}/clone")
async def clone_environment(env_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    _user(principal)
    source = await session.get(EnvironmentProfile, env_id)
    if not source:
        raise HTTPException(404, "环境不存在")
    row = EnvironmentProfile(
        id=new_id("env"),
        name=f"{source.name} 修订",
        usage=source.usage,
        embodiment_env_id=source.embodiment_env_id,
        task_id=source.task_id,
        template_id=source.template_id,
        suite_id=source.suite_id,
        definition=dict(source.definition),
        content_hash=source.content_hash,
        status="draft",
        revision_of=source.id,
    )
    session.add(row)
    await session.commit()
    return _out(row)


@router.patch("/{env_id}")
async def edit_environment(
    env_id: str, body: EnvironmentEdit, session: SessionDep, principal: PrincipalDep
) -> dict:
    _user(principal)
    row = await session.get(EnvironmentProfile, env_id)
    if not row:
        raise HTTPException(404, "环境不存在")
    if row.status != "draft":
        raise HTTPException(403, "已发布环境不可原地编辑，请创建修订")
    try:
        if row.usage == "train":
            tpl = get_template(body.template_id or row.template_id or "")
            if (
                tpl.kind not in {"sft", "ppo"}
                or not tpl.ready
                or tpl.embodiment_env_id != row.embodiment_env_id
            ):
                raise ValueError("任务模板与本体接口不匹配或未就绪")
            row.template_id, row.task_id = tpl.id, tpl.task_id
            definition = training_definition(tpl, body.settings)
        else:
            suite = await session.get(EvalSuite, body.suite_id or row.suite_id)
            if (
                not suite
                or not suite.frozen
                or not suite_executable(suite)
                or suite.embodiment_env_id != row.embodiment_env_id
            ):
                raise ValueError("评估套件与本体接口不匹配")
            row.suite_id, row.task_id = suite.id, suite_task_id(suite)
            definition = evaluation_definition(suite, body.settings)
        definition = validate_definition(row.usage, row.template_id or row.suite_id or "", definition)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not body.name.strip():
        raise HTTPException(400, "环境名称不能为空")
    row.name = body.name.strip()
    row.definition = definition
    row.content_hash = profile_hash(row)
    await session.commit()
    return _out(row)


@router.post("/{env_id}/publish")
async def publish_environment(env_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    _user(principal)
    row = await session.get(EnvironmentProfile, env_id)
    if not row:
        raise HTTPException(404, "环境不存在")
    if row.status != "draft":
        raise HTTPException(409, "只有草稿可发布")
    try:
        validate_definition(row.usage, row.template_id or row.suite_id or "", row.definition)
        if row.usage == "train":
            tpl = get_template(row.template_id or "")
            valid = (
                tpl.kind in {"sft", "ppo"}
                and tpl.ready
                and tpl.embodiment_env_id == row.embodiment_env_id
                and tpl.task_id == row.task_id
            )
        else:
            suite = await session.get(EvalSuite, row.suite_id)
            valid = bool(
                suite
                and suite.frozen
                and suite_executable(suite)
                and suite.embodiment_env_id == row.embodiment_env_id
                and suite_task_id(suite) == row.task_id
            )
    except (KeyError, ValueError):
        valid = False
    if not valid:
        raise HTTPException(400, "绑定的模板或套件与环境不兼容")
    row.content_hash = profile_hash(row)
    row.status = "active"
    await session.commit()
    return _out(row)
