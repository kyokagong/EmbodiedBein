from pathlib import Path


def test_rlinf_channel_failure_is_propagated_to_waiters():
    patch = (Path(__file__).parents[3]
             / "deploy/patches/rlinf/robotwin-single-policy-eval.patch").read_text()

    assert "rlinf/scheduler/collective/async_work.py" in patch
    assert "self._future.set_exception(exc)" in patch
