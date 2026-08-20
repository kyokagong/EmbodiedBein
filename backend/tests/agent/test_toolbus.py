from eb.harness.toolbus import FORBIDDEN, ToolBus
from eb.agent import tools as _tools  # noqa: F401


def test_forbidden_tools_blocked():
    bus = ToolBus()
    for name in FORBIDDEN:
        d = bus.check(name, {})
        assert d.allowed is False


def test_permission_matrix_is_generated():
    bus = ToolBus()
    matrix = bus.permission_matrix()
    names = {r["name"] for r in matrix}
    assert "capacity.estimate" in names
    assert "deploy.release" in names
    assert any(r["effect"] == "forbidden" for r in matrix if r["name"] == "deploy.release")


def test_submit_train_needs_approval_when_multi_gpu():
    bus = ToolBus()
    d = bus.check("job.submit.train", {"gpus": 2, "max_hours": 2})
    assert d.needs_approval is True
