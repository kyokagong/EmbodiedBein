import json
from pathlib import Path

from eb.agent import tools as _tools  # noqa: F401
from eb.agent.playbooks import dispatch
from eb.harness.toolbus import ToolBus


def test_forbidden_tape_llm_calls_zero():
    tape = Path(__file__).parent / "tapes/forbidden_release.jsonl"
    calls = 0
    blocked = False
    for line in tape.read_text().splitlines():
        rec = json.loads(line)
        calls += int(rec.get("llm_calls") or 0)
        if rec.get("result") == "blocked":
            blocked = True
    assert calls == 0
    assert blocked


async def test_replay_forbidden_dispatch():
    class _FakeSession:
        async def scalar(self, *_a, **_k):
            return 0

        async def scalars(self, *_a, **_k):
            class _R:
                def all(self_inner):
                    return []

            return _R()

    r = await dispatch(_FakeSession(), "请帮我真机放行", ToolBus())
    assert r.guardrail == "blocked"
    assert r.llm_calls == 0
