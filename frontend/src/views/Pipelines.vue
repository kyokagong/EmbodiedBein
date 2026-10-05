<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api } from "../api/client";
import EmptyState from "../components/EmptyState.vue";

const rows = ref<any[]>([]);
const jobs = ref<Record<string, any>>({});
const pending = ref(new Set<string>());
const resumeMemory = ref<Record<string, string | number>>({});
const err = ref("");
const updated = ref("");
const controller = new AbortController();
let timer: ReturnType<typeof setTimeout> | undefined;
let loading = false;

async function load() {
  if (loading || controller.signal.aborted) return;
  loading = true;
  try {
    const next = (await api("/pipelines", { signal: controller.signal })).items || [];
    const ids = [...new Set<string>(next.flatMap((p: any) =>
      (p.runs || []).flatMap((r: any) => r.job_ids || [])))];
    const details = await Promise.allSettled(ids.map(id => api(`/jobs/${id}`, { signal: controller.signal })));
    if (controller.signal.aborted) return;
    const failures: string[] = [];
    details.forEach((result, i) => {
      if (result.status === "fulfilled") jobs.value[ids[i]] = result.value;
      else failures.push(ids[i]);
    });
    rows.value = next;
    err.value = failures.length ? `作业状态读取失败：${failures.join(", ")}` : "";
    updated.value = new Date().toLocaleTimeString();
  } catch (e: any) {
    if (!controller.signal.aborted) err.value = e.message;
  } finally {
    loading = false;
  }
}

async function poll() {
  await load();
  if (!controller.signal.aborted) timer = setTimeout(poll, 5000);
}

async function start(id: string, resume = false) {
  if (pending.value.has(id)) return;
  pending.value.add(id);
  err.value = "";
  const key = Array.from(crypto.getRandomValues(new Uint32Array(4)), n => n.toString(16)).join("-");
  try {
    await api(resume ? `/pipeline-runs/${id}/resume` : `/pipelines/${id}/start`, {
      method: "POST", headers: { "Idempotency-Key": key }, signal: controller.signal,
      ...(resume && String(resumeMemory.value[id] ?? "").trim() ? { body: JSON.stringify({ memory_gb: Number(resumeMemory.value[id]) }) } : {}),
    });
    await load();
  } catch (e: any) {
    if (!controller.signal.aborted) err.value = e.message;
  } finally {
    pending.value.delete(id);
  }
}

function spec(step: any) { return step.job || step; }
function binding(step: any) {
  const source = step.bindings?.["inputs.base_model"];
  return source ? `阶段 ${source.from_step + 1} 的完整模型目录 → inputs.base_model` : "";
}

onMounted(poll);
onUnmounted(() => { controller.abort(); clearTimeout(timer); });
</script>
<template>
  <div>
    <h2>流水线</h2>
    <p class="muted">每次开始创建独立运行。下一阶段等待前一步成功，模型绑定需完整 checkpoint 回执。</p>
    <p class="muted xs">每 5 秒刷新 · {{ updated ? `最近更新 ${updated}` : "正在读取" }}</p>
    <p class="error" role="alert">{{ err }}</p>
    <EmptyState v-if="!rows.length" title="无流水线">尚无已登记的流水线模板。</EmptyState>
    <section v-for="pipeline in rows" :key="pipeline.id" class="panel" style="margin-bottom:1rem">
      <div class="row">
        <h3>{{ pipeline.name }}</h3>
        <span class="badge">{{ pipeline.status }}</span>
        <button :disabled="pending.has(pipeline.id)" @click="start(pipeline.id)">
          {{ pending.has(pipeline.id) ? "正在提交…" : "开始新运行" }}
        </button>
      </div>
      <p class="mono xs muted">{{ pipeline.id }} · 当前模板</p>
      <ol>
        <li v-for="(step, index) in pipeline.spec?.steps || []" :key="index" style="margin-bottom:.6rem">
          <strong>{{ spec(step).name || spec(step).kind }}</strong> · {{ spec(step).kind }}
          <span class="muted"> · GPU {{ spec(step).resources?.gpus ?? 0 }} · 内存 {{ spec(step).resources?.memory_gb ?? 8 }} GiB</span>
          <div v-if="binding(step)" class="xs">{{ binding(step) }}</div>
          <div v-else-if="spec(step).inputs?.base_model" class="mono xs">基础模型：{{ spec(step).inputs.base_model }}</div>
        </li>
      </ol>
      <p v-if="!pipeline.runs?.length" class="muted">尚未启动。</p>
      <details v-for="(run, runIndex) in pipeline.runs || []" :key="run.id" :open="runIndex === 0 || run.status === 'running'">
        <summary>{{ run.id }} · {{ run.status }} · {{ run.created_at }}</summary>
        <p v-if="run.resumed_from" class="muted xs">续自 {{ run.resumed_from }}，保留连续成功阶段。</p>
        <label v-if="run.status === 'failed'" class="xs">失败阶段内存上限（GiB，可选）
          <input v-model="resumeMemory[run.id]" type="number" min="1" max="1024" step="1" placeholder="留空沿用原值" :disabled="pending.has(run.id)" />
        </label>
        <button v-if="run.status === 'failed'" :disabled="pending.has(run.id)" @click="start(run.id, true)">
          {{ pending.has(run.id) ? "正在提交…" : "从失败阶段继续" }}
        </button>
        <p v-if="run.error" class="error">{{ run.error }}</p>
        <ol>
          <li v-for="jid in run.job_ids || []" :key="jid" style="margin:.6rem 0">
            <strong>{{ jobs[jid]?.kind || jid }}</strong> · {{ jobs[jid]?.status || "正在读取" }}
            <span v-if="jobs[jid]" class="muted"> · GPU {{ jobs[jid].spec?.resources?.gpus ?? 0 }} · 内存 {{ jobs[jid].spec?.resources?.memory_gb ?? 8 }} GiB</span>
            <div class="mono xs muted">{{ jid }}</div>
            <div v-if="jobs[jid]?.spec?.inputs?.base_model" class="mono xs">实际输入模型：{{ jobs[jid].spec.inputs.base_model }}</div>
            <p v-if="jobs[jid]?.error" class="error">{{ jobs[jid].error }}</p>
            <div v-for="attempt in jobs[jid]?.runs || []" :key="attempt.id" class="row xs">
              <router-link :to="`/train/runs/${attempt.id}`">Run {{ attempt.attempt }} · 日志 / 指标 · {{ attempt.id }}</router-link>
              <span>{{ attempt.status }}</span><span v-if="attempt.abort_reason" class="error">{{ attempt.abort_reason }}</span>
            </div>
            <span v-if="jobs[jid] && !jobs[jid].runs?.length" class="muted xs">尚未分配 Run，等待调度。</span>
          </li>
        </ol>
        <p v-if="run.status === 'running'" class="muted xs">后续阶段等待当前作业和输出回执；任务容器日志见上方 Run 链接。</p>
      </details>
    </section>
  </div>
</template>
