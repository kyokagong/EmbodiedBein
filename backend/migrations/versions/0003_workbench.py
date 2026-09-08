"""workbench tables and embodiment/dataset columns

Revision ID: 0003_workbench
Revises: 0002_embodiment_env
Create Date: 2026-08-30
"""

from alembic import op
from sqlalchemy import inspect, text

from eb.store.models import Base

revision = "0003_workbench"
down_revision = "0002_embodiment_env"
branch_labels = None
depends_on = None

_EMB_COLS = {
    "root_path": "text",
    "error_message": "text",
    "default_urdf": "text",
    "urdf_variants": "jsonb default '[]'::jsonb",
    "joint_tree": "jsonb default '[]'::jsonb",
    "gym_ids": "jsonb default '[]'::jsonb",
    "interface_snapshot": "jsonb",
    "scanned_at": "timestamptz",
    "updated_at": "timestamptz default now()",
}

_DS_COLS = {
    "source_kind": "text default 'path'",
    "source_uri": "text default ''",
    "dest_path": "text default ''",
    "status": "text default 'ready'",
    "total_bytes": "bigint default 0",
    "download_progress": "jsonb default '{}'::jsonb",
    "summary_stats": "jsonb default '{}'::jsonb",
    "error_message": "text",
    "updated_at": "timestamptz default now()",
}


def _col_names(insp, table: str) -> set[str]:
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "embodiments" in insp.get_table_names():
        have = _col_names(insp, "embodiments")
        for name, decl in _EMB_COLS.items():
            if name not in have:
                op.execute(text(f"ALTER TABLE embodiments ADD COLUMN {name} {decl}"))
        op.execute(
            text(
                "UPDATE embodiments SET root_path = source->>'path' "
                "WHERE root_path IS NULL AND source ? 'path'"
            )
        )
    insp = inspect(bind)
    if "datasets" in insp.get_table_names():
        have = _col_names(insp, "datasets")
        for name, decl in _DS_COLS.items():
            if name not in have:
                op.execute(text(f"ALTER TABLE datasets ADD COLUMN {name} {decl}"))
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    pass
