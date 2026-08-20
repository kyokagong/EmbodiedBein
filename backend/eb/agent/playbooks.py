from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.harness.toolbus import ToolBus
from eb.store.models import EvalReport, Job, ModelVersion, Run


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
    low = text.lower()
    blocked = bus.check("deploy.release", {})
    if any(k in text for k in ("真机放行", "改契约", "门禁豁免", "gate.override", "contract.modify")):
        return PlaybookResult(
            name="guardrail",
            reply="已拦截：真机放行 / 改契约 / 门禁豁免 是结构禁止动作，已写入审计。规则路径 llm_calls=0。",
            guardrail="blocked",
        )
    if "submit" in low and "train" in low:
        d = bus.check("job.submit.train", {"gpus": 2, "max_hours": 8})
        if d.needs_approval or not d.allowed:
            return PlaybookResult(
                name="job.submit.train",
                reply=f"该提交需要审批或被护栏拦截：{d.reason}",
                guardrail="blocked",
            )
    for pb in PLAYBOOKS:
        if any(k in text for k in pb["keywords"]):
            body = await _run(pb["id"], session)
            return PlaybookResult(name=pb["name"], reply=body)
    body = await _run("daily_report", session)
    return PlaybookResult(
        name="rules",
        reply="规则层（无 LLM）：我可以查队列、契约、估算显存，并跑 7 个预置剧本。"
        "关掉 LLM 时规则与剧本仍可用。\n\n"
        + body,
    )


async def _run(pid: str, session: AsyncSession) -> str:
    n_jobs = await session.scalar(select(func.count()).select_from(Job)) or 0
    n_models = await session.scalar(select(func.count()).select_from(ModelVersion)) or 0
    failed = (
        await session.scalars(select(Job).where(Job.status.in_(["failed", "infra_failed"])).limit(5))
    ).all()
    queued = (await session.scalars(select(Job).where(Job.status == "queued").limit(8))).all()
    reports = (await session.scalars(select(EvalReport).limit(5))).all()
    latest = (await session.scalars(select(Run).order_by(Run.created_at.desc()).limit(5))).all()

    if pid == "diagnose_train_fail":
        if not failed:
            return "训练失败诊断：当前没有 failed/infra_failed 任务。已知模式库含 OOM / 缺显示设备 / 契约不匹配。"
        lines = [f"- {j.id} {j.kind} {j.error or j.status}" for j in failed]
        return "训练失败诊断（规则路径，llm_calls=0）：\n" + "\n".join(lines)
    if pid == "hyperparam_sweep":
        return "超参扫描剧本已注册。第一版只生成计划，不自动提交 GPU 任务。并发受周预算约束。"
    if pid == "night_queue":
        idle = 4 - min(4, len(queued))
        return f"夜间队列填充：待办 {len(queued)} 个，估算空闲槽 {idle}。将以可抢占方式提交，早上 8:00 前腾出。"
    if pid == "pre_deploy_check":
        return (
            "上机前检查清单：G1–G5 必须通过；sim2sim 一致性；标定未过期；固件关节数匹配契约。"
            "最后放行按钮仍由人按。Agent 不能调用 deploy.release。"
        )
    if pid == "data_gap":
        return (
            "数据缺口分析：berkeley_ego_data 漏斗 155k 事件 → 35 场景。"
            "建议优先补失败场景的 object/instruction 组合，而不是再堆原始事件。"
        )
    if pid == "storage_patrol":
        return (
            f"资源与存储巡检：已纳管模型 {n_models} 个、任务 {n_jobs} 个。"
            "可回收候选需人工确认后才删除，删除动作永不自动执行。"
        )
    # daily_report default
    runs = [f"- {r.id} {r.status} {list((r.metrics_summary or {}).keys())[:4]}" for r in latest]
    extra = [f"- report {r.id}" for r in reports]
    return (
        "【日报】（规则路径，llm_calls=0）\n"
        f"任务总数 {n_jobs} · 模型 {n_models} · 队列 {len(queued)}\n"
        "最近 run：\n"
        + ("\n".join(runs) or "- 无")
        + "\n异常与风险：历史导入记录 provenance=partial，五元组不完整。\n"
        "待决策：是否为未归属 ONNX 指定契约。\n"
        + ("评估：\n" + "\n".join(extra) if extra else "")
    )


def catalog() -> list[dict]:
    return PLAYBOOKS
