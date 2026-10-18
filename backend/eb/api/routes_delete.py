from fastapi import APIRouter

from eb.api.deps import PrincipalDep, SessionDep
from eb.orchestrator.deletion import delete_object

router = APIRouter()


@router.get("/models/{mid}/deletion-preview")
@router.get("/weights/{mid}/deletion-preview")
async def preview_model_deletion(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "model", mid, preview=True)


@router.delete("/models/{mid}")
@router.delete("/weights/{mid}")
async def delete_model(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "model", mid)


@router.get("/train/{tid}/deletion-preview")
async def preview_train_deletion(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "train", tid, preview=True)


@router.delete("/train/{tid}")
async def delete_train(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "train", tid)


@router.get("/evals/jobs/{jid}/deletion-preview")
async def preview_eval_deletion(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "eval", jid, preview=True)


@router.delete("/evals/jobs/{jid}")
async def delete_eval(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "eval", jid)


@router.get("/jobs/{jid}/deletion-preview")
async def preview_job_deletion(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "job", jid, preview=True)


@router.delete("/jobs/{jid}")
async def delete_job(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await delete_object(session, principal, "job", jid)
