import pytest
from test_ufo_resume_preview import resume_context

from eb.api.routes_train import ResumeTrainIn, build_resume_spec, native_resume_preview_hash


def test_single_100000_resume_can_explicitly_renew_its_wall_clock_budget(monkeypatch):
    (_, _, source, checkpoint), _ = resume_context(monkeypatch)
    original = source.model_dump(mode="json")
    body = ResumeTrainIn(
        name="direct100000",
        target_updates=100000,
        native_experiment_target_updates=100000,
        timeout_seconds=21600,
        native_new_budget=True,
    )
    result = build_resume_spec(source, "trn_source", checkpoint, body)
    plan = result.params["native_train_plan"]
    assert plan["target_updates"] == plan["experiment_target_updates"] == 100000
    assert plan["segment_timeout_seconds"] == 21600 and result.policy.timeout_h == 6
    assert plan["budget_renewal"]["source_train_job_id"] == "trn_source"
    assert plan["budget_renewal"]["source_timeout_seconds"] == 20011
    assert plan["identity"] == source.params["native_train_plan"]["identity"]
    assert source.model_dump(mode="json") == original
    shorter = build_resume_spec(
        source, "trn_source", checkpoint, body.model_copy(update={"timeout_seconds": 21000})
    )
    assert native_resume_preview_hash(shorter) != native_resume_preview_hash(result)


def test_budget_cannot_expand_without_explicit_renewal(monkeypatch):
    (_, _, source, checkpoint), _ = resume_context(monkeypatch)
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_resume_spec(
            source,
            "trn_source",
            checkpoint,
            ResumeTrainIn(name="direct", target_updates=100000, timeout_seconds=21600),
        )


def test_renewal_requires_explicit_duration_and_schema_cap(monkeypatch):
    from pydantic import ValidationError

    (_, _, source, checkpoint), _ = resume_context(monkeypatch)
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_resume_spec(
            source,
            "trn_source",
            checkpoint,
            ResumeTrainIn(name="direct", target_updates=100000, native_new_budget=True),
        )
    with pytest.raises(ValidationError):
        ResumeTrainIn(name="direct", target_updates=100000, timeout_seconds=21601, native_new_budget=True)
    with pytest.raises(ValidationError):
        ResumeTrainIn(name="direct", target_updates=100000, timeout_seconds=21600, native_new_budget="yes")
