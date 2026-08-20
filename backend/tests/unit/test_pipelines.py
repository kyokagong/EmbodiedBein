from eb.domain.fsm import JobStatus
from eb.pipelines import pipeline_next


def test_empty_pipeline_succeeds():
    assert pipeline_next([], [], None) == ("succeeded", None)


def test_first_step_submits():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, [], None) == ("submit", 0)


def test_waits_while_running():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.RUNNING.value) == ("wait", None)


def test_second_step_after_success():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.SUCCEEDED.value) == ("submit", 1)


def test_fails_when_previous_failed():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.FAILED.value) == ("failed", None)
    assert pipeline_next(steps, ["job_a"], JobStatus.INFRA_FAILED.value) == ("failed", None)


def test_all_steps_done():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a", "job_b"], JobStatus.SUCCEEDED.value) == ("succeeded", None)
