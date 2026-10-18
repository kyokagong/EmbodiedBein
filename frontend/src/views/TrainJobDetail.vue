<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
import DeleteButton from "../components/DeleteButton.vue";

const route = useRoute();
const router = useRouter();
const job = ref<any>(null);
const ckpts = ref<any[]>([]);
const metrics = ref<any>({ series: {} });
const logs = ref<string[]>([]);
const err = ref("");
const notice = ref("");
const modelName = ref("");
const runId = ref("");
const acting = ref(false);
const resumeCandidate = ref<any>({ available: false });
const resumeOpen = ref(false);
const resumeName = ref("");
const resumeModelName = ref("");
const resumeTarget = ref<number | null>(null);
const resumePreview = ref<any>(null);
const resumeBusy = ref(false);
const resumeKey = ref("");
function metricPriority(name: string) {
  if (name.startsWith("train/") || name.startsWith("ufo/")) return name.includes("loss") ? 0 : 1;
  return name.startsWith("time/") ? 3 : 2;
}
const metricNames = computed(() => Object.keys(metrics.value?.series || {})
  .filter(name => Array.isArray(metrics.value.series[name]))
  .sort((a, b) => metricPriority(a) - metricPriority(b) || a.localeCompare(b)));
const singleSample = computed(() => metricNames.value.length > 0 &&
  metricNames.value.every(name => metrics.value.series[name].length === 1));
const isSft = computed(() => job.value?.kind === "sft" || job.value?.template_id?.endsWith("_sft"));
const isRpo = computed(() => ["rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo"].includes(job.value?.template_id));
const isRpoFlat = computed(() => job.value?.template_id === "rpo_v2_flat_ppo");
const latestMetric = (name: string) => {
  const points = metrics.value?.series?.[name] || [];
  return points.length ? points[points.length - 1].value : null;
};
let timer: number | undefined;
let controller: AbortController | undefined;
let generation = 0;
let disposed = false;
onUnmounted(() => { disposed = true; generation++; clearTimeout(timer); controller?.abort(); });
watch(() => route.params.id, () => {
  job.value = null;
  ckpts.value = [];
  metrics.value = { series: {} };
  logs.value = [];
  runId.value = "";
  notice.value = "";
  modelName.value = "";
  resumeCandidate.value = { available: false };
  resumeOpen.value = false;
  resumePreview.value = null;
  void load();
}, { immediate: true });

async function load() {
  clearTimeout(timer);
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  const current = ++generation;
  const id = route.params.id as string;
  try {
    const row = await api(`/train/${id}`, { signal });
    if (current !== generation) return;
    job.value = row;
    if (!modelName.value) modelName.value = row.params?.model_name || row.name;
    if (row.kind === "eval" && row.job_id) { await router.replace(`/eval/jobs/${row.job_id}`); return; }
    err.value = "";
    if (runId.value !== (row.run_id || "")) {
      metrics.value = { series: {} };
      logs.value = [];
      ckpts.value = [];
    }
    runId.value = row.run_id || "";
    const requests = [
      api(`/train/${id}/checkpoints`, { signal }),
      api(`/train/${id}/resume-candidate`, { signal }),
      runId.value ? api(`/runs/${runId.value}/metrics`, { signal }) : Promise.resolve({ series: {} }),
      runId.value ? api(`/runs/${runId.value}/logs?tail=200`, { signal }) : Promise.resolve({ lines: [] }),
    ];
    const results = await Promise.allSettled(requests);
    if (current !== generation) return;
    const failures: string[] = [];
    results.forEach((result, index) => {
      if (result.status === "rejected") {
        failures.push(`${["Checkpoint", "恢复候选", "指标", "日志"][index]}: ${result.reason.message}`);
        return;
      }
      if (result.value.error) failures.push(`${["Checkpoint", "恢复候选", "指标", "日志"][index]}: ${result.value.error}`);
      if (index === 0) ckpts.value = result.value.items || [];
      if (index === 1) resumeCandidate.value = result.value;
      if (index === 2) metrics.value = result.value;
      if (index === 3) logs.value = result.value.lines || [];
    });
    err.value = failures.join("；");
  } catch (e: any) {
    if (current === generation && e.name !== "AbortError") err.value = e.message;
  } finally {
    if (!disposed && current === generation) timer = window.setTimeout(load, 8000);
  }
}

async function register(p: any) {
  acting.value = true;
  err.value = notice.value = "";
  const current = generation;
  try {
    const row = await api(`/train/${job.value.id}/register-model`, { method: "POST", body: JSON.stringify({ path: p.path, name: modelName.value.trim() }) });
    if (current === generation) notice.value = `已登记 ${row.name} · ${row.id}`;
  } catch (e: any) { if (current === generation) err.value = e.message; }
  finally { acting.value = false; }
}
async function evaluateCheckpoint(p: any) {
  await router.push({ path: "/eval/new", query: { run_id: p.run_id, step: String(p.step) } });
}
async function cancel() {
  if (!job.value?.job_id) return;
  acting.value = true;
  const id = route.params.id;
  try {
    await api(`/jobs/${job.value.job_id}/cancel`, { method: "POST", body: "{}" });
    if (!disposed && id === route.params.id) await load();
  } catch (e: any) { if (!disposed && id === route.params.id) err.value = e.message; }
  finally { acting.value = false; }
}
function openResume() {
  if (!resumeCandidate.value?.available) return;
  resumeName.value = `${job.value.name}-resume-${resumeCandidate.value.step}`.slice(0, 120);
  resumeModelName.value = `${job.value.params?.model_name || job.value.name}-resume-${resumeCandidate.value.step}`.slice(0, 120);
  resumeTarget.value = resumeCandidate.value.suggested_target;
  resumePreview.value = null;
  resumeKey.value = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  resumeOpen.value = true;
}
function resumeBody() {
  const body: any = { name: resumeName.value, model_name: resumeModelName.value };
  body[resumeCandidate.value.termination] = resumeTarget.value;
  return body;
}
async function previewResume() {
  resumeBusy.value = true;
  err.value = "";
  try {
    resumePreview.value = await api(`/train/${job.value.id}/resume/preview`, {
      method: "POST", body: JSON.stringify(resumeBody()),
    });
  } catch (e: any) { err.value = e.message; }
  finally { resumeBusy.value = false; }
}
async function submitResume() {
  resumeBusy.value = true;
  err.value = "";
  try {
    const row = await api(`/train/${job.value.id}/resume`, {
      method: "POST", headers: { "Idempotency-Key": resumeKey.value },
      body: JSON.stringify(resumeBody()),
    });
    await router.push(`/train/${row.id}`);
  } catch (e: any) { err.value = e.message; }
  finally { resumeBusy.value = false; }
}
function st(s: string) {
  if (["running", "preparing"].includes(s)) return "st-badge st-run";
  if (s === "succeeded") return "st-badge st-ok";
  if (s === "queued") return "st-badge st-q";
  if (["failed", "aborted", "cancelled", "timeout", "infra_failed"].includes(s)) return "st-badge st-err";
  return "st-badge";
}
function gb(n: number) { return n ? n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB" : "—"; }
</script>
<template>
  <div>
    <p class="error" role="alert">{{ err }}</p>
    <p v-if="!job && !err" role="status">正在加载作业…</p>
    <template v-if="job">
    <div class="page-head">
      <div>
        <p class="crumb">训练 / {{ job.id }}</p>
        <h2>{{ job.name }} <span :class="st(job.status)">{{ job.status }}</span></h2>
      </div>
      <div class="row">
        <button v-if="job.job_id && ['queued','running','preparing'].includes(job.status)" type="button" :disabled="acting" @click="cancel">终止</button>
        <DeleteButton :path="`/train/${job.id}`" :name="job.name" :disabled="acting || !['succeeded','failed','timeout','cancelled','infra_failed'].includes(job.status)" @deleted="router.push('/train')" />
        <button v-if="resumeCandidate.available" class="primary" type="button" :disabled="resumeBusy" @click="openResume">从 checkpoint 恢复</button>
        <router-link v-if="runId" class="badge" :to="`/train/runs/${runId}`">Run 详情</router-link>
      </div>
    </div>
    <p role="status">{{ notice }}</p>
    <p class="muted xs">产出模型名称：{{ job.params?.model_name || job.name }}</p>
    <p class="error" v-if="job.error_message">{{ job.error_message }}</p>
    <p class="muted xs mono">
      镜像 {{ job.image_digest }} · 独立容器网络 · Run {{ runId || "尚未分配" }} · 输出 {{ job.run_dir || "尚未分配" }}
    </p>
    <p v-if="job.resumed_from" class="muted xs">
      恢复自 <router-link :to="`/train/${job.resumed_from}`">{{ job.resumed_from }}</router-link>
      · step {{ job.resume_step }} · {{ job.resume_mode }}
    </p>
    <p v-if="job.params?.warm_start" class="muted xs">
      初始化来源（warm start）<router-link :to="`/models/${job.params.warm_start.model_id}`">{{ job.params.warm_start.model_id }}</router-link>
      · 源 step {{ job.params.warm_start.source_step }} · 保留权重与归一化，优化器和随机状态重新初始化；本页 step 为新增训练轮数。
    </p>
    <p v-if="job.metrics_summary?.eval_report_id">评估报告：<router-link :to="`/eval/reports/${job.metrics_summary.eval_report_id}`">{{ job.metrics_summary.eval_report_id }}</router-link></p>
    <form v-if="resumeOpen" class="panel" @submit.prevent="submitResume">
      <div class="page-head">
        <div><h3>从最新完整 checkpoint 恢复</h3><p class="muted xs">源 Run {{ resumeCandidate.source_run_id }} · step {{ resumeCandidate.step }} · {{ resumeCandidate.mode }}</p></div>
        <button type="button" @click="resumeOpen = false">关闭</button>
      </div>
      <p v-if="resumeCandidate.mode === 'trainer_state'" class="muted">训练状态将恢复，仿真环境和未提交 rollout 会在 episode 边界重新初始化。</p>
      <div class="grid2">
        <label>新任务名称<input v-model="resumeName" maxlength="120" required></label>
        <label>产出模型名称<input v-model="resumeModelName" maxlength="120" required></label>
        <label>{{ resumeCandidate.termination }}<input v-model.number="resumeTarget" type="number" min="1" required></label>
      </div>
      <p class="muted xs">模板、数据、本体环境、基座模型、镜像、GPU、batch、FSDP 和 seed 均继承源任务并锁定。</p>
      <details><summary>状态覆盖</summary><pre class="preview-box">{{ JSON.stringify(resumeCandidate.state_coverage, null, 2) }}</pre></details>
      <div class="row" style="margin-top:.8rem">
        <button type="button" :disabled="resumeBusy" @click="previewResume">{{ resumeBusy ? '校验中…' : '预览恢复配置' }}</button>
        <button class="primary" type="submit" :disabled="resumeBusy || !resumePreview">{{ resumeBusy ? '提交中…' : '创建派生训练' }}</button>
      </div>
      <pre v-if="resumePreview" class="preview-box">{{ JSON.stringify(resumePreview, null, 2) }}</pre>
    </form>
    <div class="grid2">
      <div class="panel">
        <h3>{{ isSft ? '监督微调指标（loss）' : isRpoFlat ? '训练采样指标（reward / episode / throughput）' : '运行指标（reward / success / throughput）' }}</h3>
        <p v-if="singleSample" class="muted">当前 Run 每项指标仅有 1 个采样点，图上只能显示一个点，无法形成趋势曲线。</p>
        <p v-if="isRpoFlat" class="muted">env/success_once 表示轨迹中曾出现正奖励，不能当作速度／转向任务通过率。模型是否达标以独立的 100 例评估为准。</p>
        <p v-if="isRpo" class="muted">
          env/episode_len 是 Isaac 训练采样中已结束轨迹的均值，不代表冻结模型在 MuJoCo 的可运行步数。
          <span v-if="latestMetric('eval/episode_len') != null">本作业内置验证：{{ latestMetric('eval/episode_len') }} 步 / {{ latestMetric('eval/num_trajectories') }} 条已结束轨迹。</span>
        </p>
        <div v-for="name in metricNames" :key="`${runId}:${name}`">
          <Chart :points="metrics.series[name]" :title="name" />
          <p class="muted xs">{{ name }} · {{ metrics.series[name].length }} 个采样点<span v-if="metrics.series[name].length"> · 最新 {{ metrics.series[name][metrics.series[name].length - 1].value }}</span></p>
        </div>
        <p class="muted xs" v-if="!metricNames.length">尚无实测指标。仅展示运行返回的原始指标名称，不推算 loss 或 reward。</p>
      </div>
      <div class="panel">
        <h3>Checkpoint</h3>
        <label class="form-label" for="register-model-name">登记模型名称</label><input id="register-model-name" class="form-input" v-model="modelName" maxlength="120" />
        <div v-for="c in ckpts" :key="c.path" class="row" style="justify-content:space-between;margin:.35rem 0">
          <div>
            <b class="mono">{{ c.name }}</b>
            <div class="muted xs">{{ gb(c.bytes) }}</div>
          </div>
          <div class="row">
            <button v-if="c.complete" type="button" :disabled="acting || !modelName.trim()" @click="register(c)">登记为模型</button>
            <button v-if="c.weights_only && job.template_id?.startsWith('libero_')" type="button" :disabled="acting || !['succeeded','failed','timeout','cancelled','infra_failed'].includes(job.status)" @click="evaluateCheckpoint(c)">在评估菜单创建任务</button>
          </div>
        </div>
        <p v-if="!ckpts.length" class="muted xs">尚无可评估或登记的 checkpoint。</p>
        <p v-if="ckpts.some(c => c.weights_only)" class="muted xs">旧 checkpoint 仅按权重评估；这不代表可精确恢复训练。少量仿真 episode 的结果不等于正式验收。</p>
        <div style="margin-top:1rem">
          <button type="button" :disabled="!resumeCandidate.available || resumeBusy" @click="openResume">从最新完整 checkpoint 恢复</button>
          <p v-if="resumeCandidate.available" class="muted xs">{{ resumeCandidate.source_run_id }} · step {{ resumeCandidate.step }} · {{ resumeCandidate.mode }}</p>
          <p v-else class="muted xs">{{ resumeCandidate.reason || '当前没有可恢复的完整 checkpoint。' }}</p>
        </div>
      </div>
    </div>
    <div class="panel">
      <h3>输入回执</h3>
      <p class="mono xs">模板 {{ job.template_id }} · 本体 {{ job.embodiment_id }}</p>
      <p>数据集 {{ (job.dataset_ids || []).join(', ') || '无' }} · 基座模型 {{ job.base_model_id || '无' }}</p>
      <pre v-if="job.inputs" class="preview-box">{{ JSON.stringify(job.inputs, null, 2) }}</pre>
    </div>
    <div class="panel">
      <h3>Hydra Overrides</h3>
      <pre class="preview-box">{{ (job.hydra_overrides || []).join("\n") }}</pre>
    </div>
    <div class="panel">
      <h3>日志</h3>
      <pre class="preview-box" style="max-height:280px">{{ logs.join("\n") || "等待容器日志…" }}</pre>
    </div>
    </template>
  </div>
</template>
