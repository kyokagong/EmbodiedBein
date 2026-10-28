<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { api } from "../api/client";

const props = defineProps<{ model: any }>();
const status = ref<any>(null);
const preview = ref<any>(null);
const action = ref<"export" | "evals">("export");
const motionId = ref(0);
const lastJob = ref<any>(null);
const busy = ref(false);
const error = ref("");
const mediaErrors = ref<Record<string, string>>({});
let key = "";
async function refresh() {
  const id = props.model.id;
  try {
    const result = await api(`/pbfm/models/${id}`);
    if (props.model.id === id) status.value = result;
  } catch (e: any) { if (props.model.id === id) error.value = e.message; }
}
async function check(kind: "export" | "evals") {
  const id = props.model.id;
  preview.value = null; error.value = ""; busy.value = true; action.value = kind;
  try {
    const result = await api(`/pbfm/models/${id}/${kind}/preview`, {
      method: "POST", body: JSON.stringify({ motion_id: motionId.value }),
    });
    if (props.model.id === id && action.value === kind) {
      preview.value = { ...result, model_id: id, action: kind };
      key = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    }
  } catch (e: any) { if (props.model.id === id) error.value = e.message; }
  finally { busy.value = false; }
}
async function submit() {
  const reviewed = preview.value;
  if (!reviewed?.hash || reviewed.model_id !== props.model.id || busy.value) return;
  busy.value = true; error.value = "";
  try {
    lastJob.value = await api(`/pbfm/models/${reviewed.model_id}/${reviewed.action}`, {
      method: "POST", headers: { "Idempotency-Key": key },
      body: JSON.stringify({ motion_id: reviewed.plan.motion_id, preview_hash: reviewed.hash }),
    });
    preview.value = null;
  } catch (e: any) { error.value = e.message; }
  finally { busy.value = false; }
}
watch(() => props.model.id, () => { preview.value = null; status.value = null; lastJob.value = null; mediaErrors.value = {}; void refresh(); });
watch(motionId, () => { preview.value = null; });
onMounted(refresh);
</script>

<template>
  <section class="panel">
    <h3>P-BFM 深度地形评估</h3>
    <p class="muted xs">从当前冻结模型派生跟踪提示和完整深度 ONNX 策略包，再独立评估地形通过、平地回归和感知干扰。恢复状态不承诺环境或随机数精确延续。</p>
    <p class="error" v-if="error">{{ error }}</p>
    <label>参考动作索引（validation） <input v-model.number="motionId" type="number" min="0" max="100000" step="1" :disabled="busy" /></label>
    <div class="actions">
      <button class="btn" :disabled="busy" @click="check('export')">预览策略包导出</button>
      <button class="btn" :disabled="busy || !status?.export_ready" @click="check('evals')">预览独立评估</button>
      <button class="btn" :disabled="busy" @click="refresh">刷新结果</button>
    </div>
    <div v-if="preview" class="panel">
      <h4>{{ action === 'export' ? '策略包导出预览' : '独立评估预览' }}</h4>
      <p class="mono xs">{{ preview.hash }}</p>
      <pre class="xs">{{ JSON.stringify(preview.plan, null, 2) }}</pre>
      <button class="btn primary" :disabled="busy" @click="submit">{{ action === 'export' ? '开始导出' : '开始评估' }}</button>
    </div>
    <p v-if="lastJob" class="muted xs">作业 {{ lastJob.id }} · {{ lastJob.status }} · <router-link to="/train/queue">查看队列</router-link></p>
    <p v-if="status?.export_artifact_id" class="mono xs">策略包 {{ status.export_artifact_id }}</p>
    <p v-if="status?.export_url"><a :href="status.export_url" download>下载完整深度策略包</a> · <a v-if="status.export_manifest_url" :href="status.export_manifest_url" target="_blank" rel="noopener">导出清单</a></p>
    <article v-for="report in status?.eval_reports || []" :key="report.id" class="pbfm-report">
      <h4>{{ report.id }} · {{ report.qualification || '未判定' }}</h4>
      <p class="muted xs">独立 validation / 感知诊断；正式验收另行冻结。</p>
      <p v-if="report.metrics"><strong>模型质量：{{ report.gate_results?.validity !== 'complete' ? '评估未完成' : report.metrics.quality_passed === true ? '通过本套件' : report.metrics.quality_passed === false ? '未通过' : '等待完整评估' }}</strong> · 有效 {{ report.metrics.valid }} / {{ report.metrics.planned }} 例</p>
      <p v-if="report.gate_results?.validity === 'complete' && report.metrics?.cases && report.metrics.normal_success_rate != null">正常深度通过 {{ Math.round(report.metrics.normal_success_rate * report.metrics.cases) }} / {{ report.metrics.cases }}；遮挡深度通过 {{ Math.round(report.metrics.occluded_success_rate * report.metrics.cases) }} / {{ report.metrics.cases }}。训练作业成功不等于模型质量达标。</p>
      <p v-if="report.metrics?.media_categories_absent?.includes('success')" class="muted xs">本次没有成功案例，保留实际失败录像。</p>
      <p v-if="report.run_id"><router-link :to="`/train/runs/${report.run_id}`">查看评估 Run</router-link></p>
      <p class="actions">
        <a v-if="report.cases_url" :href="report.cases_url" target="_blank" rel="noopener">逐例结果</a>
        <a v-if="report.media_index_url" :href="report.media_index_url" target="_blank" rel="noopener">媒体索引</a>
        <a v-if="report.plot_url" :href="report.plot_url" target="_blank" rel="noopener">分组图表</a>
      </p>
      <img v-if="report.plot_url" :src="report.plot_url" alt="P-BFM 地形评估分组图表" loading="lazy" @error="mediaErrors[report.plot_url] = '图表加载失败'" />
      <p v-if="mediaErrors[report.plot_url]" class="error" role="alert">{{ mediaErrors[report.plot_url] }}</p>
      <pre v-if="report.metrics" class="mono xs">{{ JSON.stringify(report.metrics, null, 2) }}</pre>
      <p v-if="!report.cases?.length" class="error">缺少逐例证据。</p>
      <details v-for="item in report.cases || []" :key="`${item.case_id}-${item.depth_condition}`">
        <summary>{{ item.case_id }} · {{ item.depth_condition }} · {{ item.success ? '通过' : '未通过' }}</summary>
        <video v-if="item.video_url" :src="item.video_url" controls preload="metadata" :aria-label="`${item.case_id} 仿真录像`" @error="mediaErrors[item.video_url] = '仿真录像加载失败'" />
        <p v-else class="error">缺少仿真录像。</p>
        <p v-if="mediaErrors[item.video_url]" class="error" role="alert">{{ mediaErrors[item.video_url] }}</p>
        <video v-if="item.depth_url" :src="item.depth_url" controls preload="metadata" :aria-label="`${item.case_id} 实际网络深度输入`" @error="mediaErrors[item.depth_url] = '深度录像加载失败'" />
        <p v-else class="error">缺少网络深度输入录像。</p>
        <p v-if="mediaErrors[item.depth_url]" class="error" role="alert">{{ mediaErrors[item.depth_url] }}</p>
        <a v-if="item.depth_data_url" :href="item.depth_data_url" download>原始深度历史</a>
        <a v-if="item.trace_url" :href="item.trace_url" target="_blank" rel="noopener">原始轨迹与时序</a>
      </details>
    </article>
  </section>
</template>

<style scoped>
.pbfm-report { margin-top: 1rem; border-top: 1px solid var(--line); }
details { margin: .75rem 0; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
video, img { display: block; width: 100%; max-width: 720px; margin: .6rem 0; }
</style>
