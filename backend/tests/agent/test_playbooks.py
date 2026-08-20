from eb.agent import tools as _tools  # noqa: F401
from eb.agent.playbooks import PLAYBOOKS, dispatch
from eb.harness.toolbus import ToolBus


class _FakeSession:
    async def scalar(self, *_a, **_k):
        return 0

    async def scalars(self, *_a, **_k):
        class _R:
            def all(self_inner):
                return []

        return _R()


async def test_seven_playbooks_registered():
    assert len(PLAYBOOKS) == 7


async def test_forbidden_actions_blocked_without_llm():
    r = await dispatch(_FakeSession(), "请帮我真机放行并把契约改掉", ToolBus())
    assert r.guardrail == "blocked"
    assert r.llm_calls == 0


async def test_daily_report_llm_calls_zero():
    r = await dispatch(_FakeSession(), "写一份日报", ToolBus())
    assert r.llm_calls == 0
    assert "日报" in r.reply or r.name == "日报"
