"""embodiment / env / task tables; rename leftover contract columns

Revision ID: 0002_embodiment_env
Revises: 0001_init
Create Date: 2026-08-25
"""

from alembic import op
from sqlalchemy import inspect, text

from eb.store.models import Base

revision = "0002_embodiment_env"
down_revision = "0001_init"
branch_labels = None
depends_on = None

_RENAME_COLS = [
    ("jobs", "contract_id", "embodiment_env_id"),
    ("experiments", "contract_id", "embodiment_env_id"),
    ("datasets", "contract_id", "embodiment_env_id"),
    ("model_versions", "contract_id", "embodiment_env_id"),
    ("eval_suites", "contract_id", "embodiment_env_id"),
    ("bundles", "contract_id", "embodiment_env_id"),
    ("robots", "contract_id", "embodiment_env_id"),
]


def _col_names(insp, table: str) -> set[str]:
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "embodiment_contracts" in tables and "embodiment_envs" not in tables:
        op.rename_table("embodiment_contracts", "embodiment_envs")
        insp = inspect(bind)

    if "embodiment_envs" in insp.get_table_names() and "embodiment_id" not in _col_names(insp, "embodiment_envs"):
        op.execute(text("ALTER TABLE embodiment_envs ADD COLUMN embodiment_id text"))
        op.execute(
            text(
                "UPDATE embodiment_envs SET embodiment_id = 'emb_berkeley_humanoid_lite' "
                "WHERE embodiment_id IS NULL"
            )
        )

    for table, old, new in _RENAME_COLS:
        cols = _col_names(insp, table)
        if old in cols and new not in cols:
            op.execute(text(f'ALTER TABLE {table} RENAME COLUMN {old} TO {new}'))

    insp = inspect(bind)
    if "dataset_versions" in insp.get_table_names():
        cols = _col_names(insp, "dataset_versions")
        if "compatible_contracts" in cols and "compatible_envs" not in cols:
            op.execute(text("ALTER TABLE dataset_versions RENAME COLUMN compatible_contracts TO compatible_envs"))

    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    pass
