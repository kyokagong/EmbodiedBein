from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from eb.ids import sha256_bytes
from eb.store.db import get_session
from eb.store.models import ApiKey

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class Principal:
    def __init__(self, name: str, actor_type: str) -> None:
        self.name = name
        self.actor_type = actor_type

    @property
    def id(self) -> str:
        return f"{self.actor_type}:{self.name}"


async def current_principal(request: Request, session: SessionDep) -> Principal:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        token_hash = sha256_bytes(token.encode())
        from sqlalchemy import select

        key = await session.scalar(select(ApiKey).where(ApiKey.token_hash == token_hash, ApiKey.revoked.is_(False)))
        if key:
            actor, _, name = key.principal.partition(":")
            return Principal(name or key.principal, key.actor_type)
        raise HTTPException(401, "invalid api key")
    user = request.session.get("user")
    if user:
        return Principal(user, "user")
    # showcase and health are excluded by not depending on this
    raise HTTPException(401, "not authenticated")


PrincipalDep = Annotated[Principal, Depends(current_principal)]
