import pytest
from pydantic import ValidationError
from test_ufo_resume_preview import resume_context
from test_ufo_training import native_spec

from eb.api.routes_train import ResumeTrainIn, build_resume_spec, native_resume_preview_hash
from eb.ufo_training import render_training, validate_train_job


def test_allocator_is_explicit_and_bound_to_resume_preview(monkeypatch):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    original = source.model_dump(mode="json")
    plain = build_resume_spec(source, "trn_source", checkpoint, body)
    body = body.model_copy(update={"native_cuda_allocator": "expandable_segments"})
    optimized = build_resume_spec(source, "trn_source", checkpoint, body)
    assert optimized.params["native_train_plan"]["cuda_allocator"] == "expandable_segments"
    assert native_resume_preview_hash(plain) != native_resume_preview_hash(optimized)
    assert optimized.params["native_train_plan"]["identity"] == plain.params["native_train_plan"]["identity"]
    assert (optimized.params["native_train_plan"]["native_args"]
            == plain.params["native_train_plan"]["native_args"])
    assert source.model_dump(mode="json") == original
    inherited = build_resume_spec(
        optimized, "trn_next", checkpoint, ResumeTrainIn(name="next", target_updates=1008)
    )
    assert inherited.params["native_train_plan"]["cuda_allocator"] == "expandable_segments"


def test_allocator_render_preserves_old_jobs_and_freezes_explicit_environment():
    spec = native_spec()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in render_training(spec, "run_preview").env
    spec.params["native_train_plan"]["cuda_allocator"] = "expandable_segments"
    assert validate_train_job(spec)
    rendered = render_training(spec, "run_preview")
    assert rendered.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert rendered.image == spec.image


@pytest.mark.parametrize("value", ["cudaMallocAsync", "arbitrary", True])
def test_allocator_rejects_unreviewed_settings(value):
    with pytest.raises(ValidationError):
        ResumeTrainIn(name="next", target_updates=1008, native_cuda_allocator=value)
    spec = native_spec()
    spec.params["native_train_plan"]["cuda_allocator"] = value
    with pytest.raises(ValueError, match="allocator"):
        validate_train_job(spec)
