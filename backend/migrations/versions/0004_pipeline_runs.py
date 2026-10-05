"""Isolate pipeline templates from execution snapshots."""
from alembic import op
from sqlalchemy import text

from eb.store.models import PipelineRun

revision = "0004_pipeline_runs"
down_revision = "0003_workbench"
branch_labels = None
depends_on = None


def upgrade():
    PipelineRun.__table__.create(op.get_bind(), checkfirst=True)
    # Keep a snapshot of the existing execution; do not rewrite historical templates.
    op.execute(text("""
        INSERT INTO pipeline_runs (id, pipeline_id, spec, status, job_ids, created_by)
        SELECT 'pr_legacy_' || id, id, spec, spec->>'status',
               COALESCE(spec->'job_ids', '[]'::jsonb), 'user:kk'
        FROM pipelines WHERE spec->>'status' IN ('running', 'succeeded', 'failed')
        ON CONFLICT (id) DO NOTHING
    """))


def downgrade():
    PipelineRun.__table__.drop(op.get_bind(), checkfirst=True)
