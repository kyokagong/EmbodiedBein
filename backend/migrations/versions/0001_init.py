"""init schema

Revision ID: 0001_init
Revises:
Create Date: 2026-08-20
"""

from alembic import op
from sqlalchemy import inspect

from eb.store.models import Base

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
