"""Claim run outputs into CAS / models. Expanded in later tasks; Task 4 is a no-op."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from eb.store.models import Job, Run


async def claim_outputs(session: AsyncSession, job: Job, run: Run) -> None:
    _ = (session, job, run)
    return
