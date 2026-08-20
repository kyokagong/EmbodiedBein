from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.store.models import LineageEdge


async def add_edge(session: AsyncSession, src_type: str, src_id: str, dst_type: str, dst_id: str, relation: str, run_id: str | None = None) -> None:
    session.add(
        LineageEdge(
            src_type=src_type,
            src_id=src_id,
            dst_type=dst_type,
            dst_id=dst_id,
            relation=relation,
            run_id=run_id,
        )
    )


async def upstream(session: AsyncSession, dst_type: str, dst_id: str) -> list[LineageEdge]:
    return list(await session.scalars(select(LineageEdge).where(LineageEdge.dst_type == dst_type, LineageEdge.dst_id == dst_id)))
