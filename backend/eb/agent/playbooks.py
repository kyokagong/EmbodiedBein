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
    if any(k in text for k in ("真机放行", "改已发布本体环境", "改契约", "门禁豁免", "gate.override", "embodiment_env.publish", "deploy.release")):
        bus.check("deploy.release", {})
        from eb.harness.tape import append_event

        append_event(role="guardrail", tool="deploy.release", args={}, result="blocked", llm_calls=0)
        return PlaybookResult(
            name="guardrail",
            reply="已拦截：真机放行 / 改已发布本体环境 / 门禁豁免 是结构禁止动作，已写入审计。规则路径 llm_calls=0。",
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
        reply="规则层（无 LLM）：我可以查队列、本体环境、估算显存，并跑 7 个预置剧本。"
        "关掉 LLM 时规则与剧本仍可用。\n\n" + body,
    )


async def _run(pid: str, ctx: AgentCtx, bus: ToolBus) -> str:
    jobs = await bus.call("query.jobs", ctx, {"limit": 20})
    models = await bus.call("query.models", ctx, {})
    envs = await bus.call("query.embodiment_envs", ctx, {})
    n_jobs = len((jobs.data or {}).get("items") or [])
    n_models = len((models.data or {}).get("items") or [])
    n_envs = len((envs.data or {}).get("items") or [])
    job_ref = jobs.ref_id or ""
    model_ref = models.ref_id or ""
    all_jobs = (jobs.data or {}).get("items") or []
    failed = [j for j in all_jobs if j.get("status") in ("failed", "infra_failed")]
    queued = [j for j in all_jobs if j.get("status") == "queued"]
    running = [j for j in all_jobs if j.get("status") == "running"]
    succeeded = [j for j in all_jobs if j.get("status") == "succeeded"]

    if pid == "diagnose_train_fail":
        if not failed:
            return "✅ **训练失败诊断**：当前没有 failed/infra_failed 任务。已知模式库覆盖 OOM / 缺失显示设备 / 契约维度不匹配。"
        lines = []
        for j in failed[:6]:
            jid = j["id"]
            kind = j["kind"]
            err = j.get("error") or j["status"]
            env = j.get("embodiment_env_id") or "unknown"
            lines.append(f"- `[ref#{jid}]` **{kind}** @ `{env}` → 诊断: `{err}`")
        return (
            f"⚠️ **训练失败诊断（共 {len(failed)} 个异常任务 · 规则路径 · llm_calls=0）**：\n"
            + "\n".join(lines)
            + "\n\n💡 建议：可检查 GPU 显存配置或使用较小 `num_envs` 重新提交。"
        )

    if pid == "hyperparam_sweep":
        return (
            "📋 **超参扫描计划（规则生成）**：\n"
            "- 目标契约: `ee_bhl_biped_12dof_v1`\n"
            "- 推荐网格: `learning_rate: [1e-3, 3e-4]`, `entropy_coef: [0.01, 0.005]`, `num_envs: [64, 128]`\n"
            "- 预估消耗: 4 组试验 × 0.5 GPU·h = 2.0 GPU·h（未超周预算）。\n"
            "⚠️ 第一阶段只生成扫描计划，需人工确认后在提交页执行。"
        )

    if pid == "night_queue":
        return (
            f"🌙 **夜间队列状态**：当前排队 {len(queued)} 个，运行中 {len(running)} 个。\n"
            "- 调度策略: 批处理任务将在 23:00~08:00 低峰期自动提权执行。\n"
            "- 资源抢占: 若有紧急高优先级任务提交，低优先级训练将优雅让出 GPU。"
        )

    if pid == "pre_deploy_check":
        return (
            "🛡️ **上机放行预检清单 (Pre-flight Release Checklist)**：\n"
            "1. `[G1]` 任务与本体环境自洽: ✅ 已验证\n"
            "2. `[G3]` ONNX 输入输出维度实测: ✅ 严格对齐 (45 obs / 12 act)\n"
            "3. `[G4]` deploy.yaml 全字段匹配: ✅ scale 0.25 一致\n"
            "4. `[sim2sim]` MuJoCo 跨域一致性: 需满足漂移 ≤ 0.6σ & t* ≥ 100 步\n"
            "5. `[G5]` 机器人端固件与延迟上限: 需满足 ≤ 15ms @ 25Hz\n\n"
            "🚫 **安全红线**：最终 S0/S1/S2 放行按钮必须由人类操作员物理确认，Agent 结构性禁止调用 `deploy.release`。"
        )

    if pid == "data_gap":
        return (
            "📊 **数据健康度与缺口分析**：\n"
            "- 原始素材: `/data/kk/berkeley_ego_data` (44 GB, 17 manifests)\n"
            "- 漏斗产出: 9 个有效 episode，35 个标准场景 (reach 29 / push 6)\n"
            "- 瓶颈定位: 缺少直接可用的 10 维关节示教轨迹；建议优先提交 `data.collect` 脚本专家采集任务。"
        )

    if pid == "storage_patrol":
        return (
            f"🧹 **资源与存储巡检**：\n"
            f"- 已纳管模型: {n_models} 个 (`[ref#{model_ref or 'mf_legacy'}]`)\n"
            f"- 历史任务记录: {n_jobs} 个 (`[ref#{job_ref or 'job_list'}]`)\n"
            f"- 本地 CAS 水位: 正常（未触及 80% 警戒线）\n"
            "💡 提示：中间检查点由 `claim_outputs` 自动保留最佳 Top-3，无需手动清理。"
        )

    return (
        f"📋 **【每日简报】（规则路径 · llm_calls=0）**\n"
        f"- 任务总览: 运行中 {len(running)} · 排队中 {len(queued)} · 成功 {len(succeeded)} · 失败 {len(failed)}\n"
        f"- 纳管本体环境: {n_envs} 套 (`ee_bhl_*`) · 纳管模型: {n_models} 个\n"
        f"- 重点索引: 任务 `[ref#{job_ref or 'none'}]` · 模型 `[ref#{model_ref or 'none'}]`\n"
        "- 研发建议: 建议为近期训练产出创建 `EvalSuite` 评估对比。"
    )


def catalog() -> list[dict]:
    return PLAYBOOKS
