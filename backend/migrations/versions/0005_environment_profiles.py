"""Operator-facing versioned train and eval environments."""

from alembic import op

from eb.store.models import EnvironmentProfile

revision = "0005_environment_profiles"
down_revision = "0004_pipeline_runs"
branch_labels = None
depends_on = None


def upgrade():
    EnvironmentProfile.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    EnvironmentProfile.__table__.drop(op.get_bind(), checkfirst=True)
