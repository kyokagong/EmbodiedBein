<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import DeleteButton from "../components/DeleteButton.vue";

const route = useRoute(); const job = ref<any>(null); const progress = ref<any>(null); const logs = ref<string[]>([]); const error = ref("");
const router = useRouter();
let timer: number | undefined; let alive = true;
onUnmounted(() => { alive = false; clearTimeout(timer); });
async function load() {
  try {
    const row = await api(`/evals/jobs/${route.params.id}`);
    if (!alive) return;
    job.value = row; error.value = "";
    if (row.plan && row.plan.schema_version !== "eb/rpo-wbc-eval-plan/v1")
      progress.value = await api(`/evals/jobs/${route.params.id}/progress`);
    if (row.run_id) logs.value = (await api(`/runs/${row.run_id}/logs?tail=100`)).lines || [];
  } catch (e: any) { if (alive) error.value = e.message; }
  finally { if (alive) timer = window.setTimeout(load, 8000); }
}
async function cancel() {
  if (!job.value) return;
  try { await api(`/jobs/${job.value.id}/cancel`, {method: "POST", body: "{}"}); await load(); }
  catch (e: any) { error.value = e.message; }
}
onMounted(load);
</script>
<template>
  <div v-if="job"><div class="page-head"><div><p class="crumb"><router-link to="/eval">评估任务</router-link> / {{ job.id }}</p><h2>{{ job.name }}</h2></div>
    <button v-if="['queued','admitted','preparing','running'].includes(job.status)" @click="cancel">取消任务</button>
    <DeleteButton :path="`/evals/jobs/${job.id}`" :name="job.name" :disabled="!['succeeded','failed','timeout','cancelled','infra_failed'].includes(job.status)" @deleted="router.push('/eval')" /></div>
    <p class="error" v-if="error">{{ error }}</p><p>执行状态：<b>{{ job.status }}</b> · 套件 {{ job.suite_id || '历史' }}</p>
    <p v-if="job.error" class="error">{{ job.error }}</p>
    <div class="grid2"><div class="panel"><h3>受评输入</h3><p v-if="job.model_version_id">模型 <router-link :to="`/models/${job.model_version_id}`">{{ job.model_version_id }}</router-link> · 源 Run {{ job.plan?.source_run_id }} / step {{ job.plan?.checkpoint_step }}</p><p v-else>源 Run {{ job.checkpoint?.run_id }} · step {{ job.checkpoint?.step }}</p>
      <p v-if="job.plan?.backend">{{ job.plan.backend === 'isaac' ? 'Isaac Lab' : 'MuJoCo' }} · {{ job.plan?.seeds?.length }} seeds · {{ job.record_video ? '录像回放' : '无画面指标评估' }}</p>
      <p class="muted xs">{{ job.plan?.hash || '历史评估：未保存完整计划' }}</p></div>
      <div class="panel"><h3>覆盖与结果</h3><p>已记录 {{ progress?.recorded ?? job.metrics_summary?.n_trials ?? '—' }} / {{ progress?.planned ?? job.plan?.planned ?? job.plan?.seeds?.length ?? '—' }} 条</p>
      <progress v-if="progress?.planned" :value="progress.recorded" :max="progress.planned" style="width:100%"></progress>
      <p v-if="progress">完成 {{ progress.percent }}%<span v-if="progress.duplicate_records || progress.invalid_records" class="error"> · 已忽略重复 {{ progress.duplicate_records }} 条、无效 {{ progress.invalid_records }} 条</span></p>
      <p v-if="job.plan?.model?.runtime">模型副本 {{ job.plan.model.runtime.policy_worker_count }} · 总并发环境 {{ job.plan.model.runtime.parallel_envs }} · 推理 micro-batch {{ job.plan.model.runtime.inference_micro_batch_size }}</p>
      <p v-if="job.plan?.model?.runtime?.resources">档位 {{ job.plan.model.runtime.profile_id }} / {{ job.plan.model.runtime.resource_variant_id }} · 容器内存 {{ job.plan.model.runtime.resources.memory_gb }} GiB · 共享内存 {{ job.plan.model.runtime.resources.shm_gb }} GiB</p>
      <p v-if="job.plan?.model?.runtime">Headless · 离屏三相机 · 视频关闭</p>
      <p>成功率 {{ job.metrics_summary?.success_rate == null ? '—' : (job.metrics_summary.success_rate * 100).toFixed(1) + '%' }}</p>
      <p v-if="job.plan?.backend">跌倒 {{ job.metrics_summary?.fall_count ?? '—' }} · 速度 RMSE {{ job.metrics_summary?.velocity_rmse ?? '—' }} · 上肢 RMSE {{ job.metrics_summary?.upper_rmse ?? '—' }}</p>
      <router-link v-if="job.metrics_summary?.eval_report_id" :to="`/eval/reports/${job.metrics_summary.eval_report_id}`">查看评估报告 →</router-link>
      <p v-else class="muted">等待报告归档；任务完成不代表模型通过质量验收。</p></div></div>
    <div v-if="progress?.by_task?.length" class="panel"><h3>分任务进度</h3><table><thead><tr><th>任务</th><th>已记录</th><th>计划</th><th>完成度</th></tr></thead><tbody>
      <tr v-for="row in progress.by_task" :key="row.task_id"><td>{{ row.task_id }}</td><td>{{ row.recorded }}</td><td>{{ row.planned }}</td><td>{{ (row.recorded * 100 / row.planned).toFixed(1) }}%</td></tr>
    </tbody></table></div>
    <p v-if="job.run_id"><router-link :to="`/eval/runs/${job.run_id}`">Run 运行详情</router-link></p>
    <div class="panel"><h3>日志</h3><pre style="max-height:300px;overflow:auto">{{ logs.join('\n') || '等待运行日志…' }}</pre></div>
  </div><p v-else-if="error" class="error">{{ error }}</p>
</template>
