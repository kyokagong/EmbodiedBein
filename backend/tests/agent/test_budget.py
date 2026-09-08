
from eb.agent import tools as _tools  # noqa: F401
from eb.agent.tools import AgentCtx
from eb.harness.toolbus import ToolBus


async def test_budget_blocks_submit_train(monkeypatch):
    monkeypatch.setenv("EB_AGENT_WEEKLY_GPU_HOURS", "0.01")
    monkeypatch.setenv("EB_AGENT_GPU_HOURS_USED", "1.0")
    bus = ToolBus()

    class _S:
        pass

    spec = {"kind": "train.rl", "embodiment_env_id": "ee_bhl_biped_12dof_v1", "task_id": "tsk_bhl_velocity_biped", "gpus": 1}
    r = await bus.call("job.submit.train", AgentCtx(api=_S(), principal="t"), spec)
    assert r.ok is False
    assert "budget" in (r.error or "")
