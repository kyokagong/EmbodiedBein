from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from eb.agent.tools import AgentCtx
from eb.harness.toolbus import ToolBus


@dataclass
class PlaybookResult:
    name: str
    reply: str
    llm_calls: int = 0
    guardrail: str = "ok"


PLAYBOOKS = [
    {
        "id": "diagnose_train_fail",
        "name": "训练失败诊断",
        "trigger": "run 进入 failed",
        "keywords": ("失败", "诊断", "oom", "failed"),
    },
    {
        "id": "hyperparam_sweep",
        "name": "超参扫描",
        "trigger": "人工发起",
        "keywords": ("超参", "扫描", "sweep"),
    },
    {
        "id": "night_queue",
        "name": "夜间队列填充",
        "trigger": "每晚 23:00",
        "keywords": ("夜间", "闲置", "填充"),
    },
    {
        "id": "pre_deploy_check",
        "name": "上机前检查",
        "trigger": "策略包创建",
        "keywords": ("上机", "放行", "sim2sim", "检查清单"),
    },
    {
        "id": "data_gap",
        "name": "数据缺口分析",
        "trigger": "评估报告产出",
        "keywords": ("缺口", "数据", "场景"),
    },
    {
        "id": "storage_patrol",
        "name": "资源与存储巡检",
        "trigger": "每天 07:00",
        "keywords": ("磁盘", "巡检", "回收"),
    },
    {
        "id": "daily_report",
        "name": "日报",
        "trigger": "每天 09:00",
        "keywords": ("日报", "汇总", "今天", "产出"),
    },
]


async def dispatch(session: AsyncSession, text: str, bus: ToolBus) -> PlaybookResult:
    ctx = AgentCtx(api=session, principal="agent:playbook")
    if any(k in text for k in ("真机放行", "改契约", "门禁豁免", "gate.override", "contract.modify", "deploy.release")):
        bus.check("deploy.release", {})
        from eb.harness.tape import append_event

        append_event(role="guardrail", tool="deploy.release", args={}, result="blocked", llm_calls=0)
        return PlaybookResult(
            name="guardrail",
            reply="已拦截：真机放行 / 改契约 / 门禁豁免 是结构禁止动作，已写入审计。规则路径 llm_calls=0。",
            guardrail="blocked",
        )
    low = text.lower()
    if "submit" in low and "train" in low:
        d = bus.check("job.submit.train", {"gpus": 2, "max_hours": 8})
        if d.needs_approval or not d.allowed:
            return PlaybookResult(
                name="job.submit.train",
                reply=f"该提交需要审批或被护栏拦截：{d.reason}",
                guardrail="blocked",
            )
        from eb.harness.budget import allow_costly

        if not allow_costly():
            return PlaybookResult(
                name="job.submit.train",
                reply="周 GPU·h 预算已用尽，只允许 effect=read 工具。",
                guardrail="blocked",
            )
    for pb in PLAYBOOKS:
        if any(k in text for k in pb["keywords"]):
            body = await _run(pb["id"], ctx, bus)
            return PlaybookResult(name=pb["name"], reply=body)
    from eb.agent.llm import maybe_llm

    llm = await maybe_llm(text)
    if llm:
        return PlaybookResult(name="llm", reply=llm, llm_calls=1)
    body = await _run("daily_report", ctx, bus)
    return PlaybookResult(
        name="rules",
        reply="规则层（无 LLM）：我可以查队列、契约、估算显存，并跑 7 个预置剧本。"
        "关掉 LLM 时规则与剧本仍可用。\n\n" + body,
    )


async def _run(pid: str, ctx: AgentCtx, bus: ToolBus) -> str:
    jobs = await bus.call("query.jobs", ctx, {"limit": 20})
    models = await bus.call("query.models", ctx, {})
    n_jobs = len((jobs.data or {}).get("items") or [])
    n_models = len((models.data or {}).get("items") or [])
    job_ref = jobs.ref_id or ""
    model_ref = models.ref_id or ""
    failed = [j for j in (jobs.data or {}).get("items") or [] if j.get("status") in ("failed", "infra_failed")]
    queued = [j for j in (jobs.data or {}).get("items") or [] if j.get("status") == "queued"]

    if pid == "diagnose_train_fail":
        if not failed:
            return "训练失败诊断：当前没有 failed/infra_failed 任务。已知模式库含 OOM / 缺显示设备 / 契约不匹配。"
        lines = [f"- {j['id']} {j['kind']} {j.get('error') or j['status']}" for j in failed[:8]]
        return "训练失败诊断（规则路径，llm_calls=0）：\n" + "\n".join(lines)
    if pid == "hyperparam_sweep":
        return "超参扫描剧本已注册。第一版只生成计划，不自动提交 GPU 任务。并发受周预算约束。"
    if pid == "night_queue":
        return f"夜间队列填充：待办 {len(queued)} 个。将以可抢占方式提交，早上 8:00 前腾出。"
    if pid == "pre_deploy_check":
        return (
            "上机前检查清单：G1–G5 必须通过；sim2sim 一致性；标定未过期；固件关节数匹配契约。"
            "最后放行按钮仍由人按。Agent 不能调用 deploy.release。"
        )
    if pid == "data_gap":
        return "数据缺口分析：漏斗数字来自最近一次 ebd scan，而不是种子 155000。"
    if pid == "storage_patrol":
        return (
            f"资源与存储巡检：已纳管模型 {n_models} 个、最近任务 {n_jobs} 个。"
            "可回收候选需人工确认后才删除，删除动作永不自动执行。"
        )
    return (
        "【日报】（规则路径，llm_calls=0）\n"
        f"最近任务 {n_jobs} · 模型 {n_models}\n"
        f"任务 ref `{job_ref or 'none'}` · 模型 ref `{model_ref or 'none'}`\n"
        "异常与风险：历史导入 provenance 可能为 partial。\n"
        "待决策：是否为未归属 ONNX 指定契约。"
    )


def catalog() -> list[dict]:
    return PLAYBOOKS
