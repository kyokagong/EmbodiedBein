<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { api } from "../api/client";

const props = defineProps<{ model?: any }>();
const emit = defineEmits<{ changed: [] }>();
type Action = "imports" | "export" | "evals" | "bundles";
const capabilities = ref<any>(null);
const status = ref<any>(null);
const sourceName = ref("UFO/RPO 原生模型");
const sourcePath = ref("");
const backend = ref("mjlab");
const profile = ref("smoke");
const preview = ref<any>(null);
const lastJob = ref<any>(null);
const busy = ref(false);
const loading = ref(false);
const error = ref("");
const mediaErrors = ref<Record<string, string>>({});
const controller = new AbortController();
let timer: ReturnType<typeof setTimeout> | undefined;
let pollsLeft = 120;
let previewVersion = 0;
const currentModel = computed(() => status.value?.model || props.model);
const exported = computed(() => currentModel.value?.card?.schema_version === "eb/rpo-ufo-native-export/v1");
const reports = computed<any[]>(() => status.value?.reports || []);
const jobs = computed<any[]>(() => {
  const rows = status.value?.jobs || [];
  return lastJob.value && !rows.some((job: any) => job.id === lastJob.value.id) ? [lastJob.value, ...rows] : rows;
});
const active = (job: any) => ["draft", "blocked", "queued", "admitted", "preparing", "running", "preempted"].includes(job?.status);
const actionLabels: Record<Action, string> = { imports: "导入", export: "导出", evals: "仿真评估", bundles: "构建仿真交付包" };

watch([sourceName, sourcePath, backend, profile], () => { previewVersion++; preview.value = null; });
onMounted(reload);
onUnmounted(() => { controller.abort(); clearTimeout(timer); });

function endpoint(action: Action) {
  return props.model ? `/ufo-native/models/${props.model.id}/${action}` : "/ufo-native/imports";
}
function runId(job: any) { return job.run_id || job.runs?.at(-1)?.id; }
function imageFile(path: string) { return /\.(svg|png|jpe?g|webp)$/i.test(path); }
function reportState(name: string) { return reports.value.find(report => report.backend === name)?.qualification || "未评估"; }

async function reload() {
  if (loading.value || busy.value) return;
  clearTimeout(timer);
  loading.value = true; error.value = "";
  try {
    const result = await api("/ufo-native/capabilities", { signal: controller.signal });
    if (controller.signal.aborted) return;
    capabilities.value = result;
    if (!sourcePath.value) sourcePath.value = result.source_default || "";
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { loading.value = false; }
  await refresh();
}
async function refresh() { pollsLeft = 120; await loadState(); }
async function loadState() {
  clearTimeout(timer);
  if (loading.value || controller.signal.aborted) return;
  if (!props.model && !lastJob.value) return;
  loading.value = true;
  try {
    const result = await api(props.model ? `/ufo-native/models/${props.model.id}/status` : `/jobs/${lastJob.value.id}`, { signal: controller.signal });
    if (controller.signal.aborted) return;
    if (props.model) {
      status.value = result;
      if (lastJob.value && result.jobs?.some((job: any) => job.id === lastJob.value.id)) lastJob.value = null;
    } else {
      const wasActive = active(lastJob.value);
      lastJob.value = result;
      if (wasActive && !active(result)) emit("changed");
    }
    if (jobs.value.some(active) && pollsLeft-- > 0) timer = setTimeout(loadState, 5000);
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { loading.value = false; }
}
async function check(action: Action) {
  if (busy.value) return;
  busy.value = true; error.value = ""; preview.value = null;
  const version = previewVersion;
  const body = action === "imports" ? { name: sourceName.value, path: sourcePath.value }
    : action === "evals" ? { backend: backend.value, profile: profile.value } : {};
  try {
    const result = await api(`${endpoint(action)}/preview`, { method: "POST", signal: controller.signal, body: JSON.stringify(body) });
    if (!controller.signal.aborted && version === previewVersion) preview.value = {
      ...result, action, body, key: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    };
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { busy.value = false; }
}
async function submit() {
  if (!preview.value?.plan?.hash || busy.value) return;
  const checked = preview.value;
  busy.value = true; error.value = "";
  try {
    const job = await api(endpoint(checked.action), { method: "POST", signal: controller.signal,
      headers: { "Idempotency-Key": checked.key }, body: JSON.stringify({ ...checked.body, preview_hash: checked.plan.hash }) });
    if (controller.signal.aborted) return;
    lastJob.value = job;
    preview.value = null;
    await refresh();
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { busy.value = false; }
}
</script>

<template>
  <section class="panel ufo-native-panel" aria-labelledby="ufo-native-title">
    <div class="page-head">
      <h3 id="ufo-native-title">{{ model ? 'UFO/RPO 原生推理与部署前仿真' : '导入 UFO/RPO 原生模型' }}</h3>
      <button type="button" :disabled="loading || busy" @click="reload">{{ loading ? '刷新中…' : '刷新状态' }}</button>
    </div>
    <p class="muted">本机容器推理与仿真交付；目标硬件未验证，交付包仅用于仿真。</p>
    <p v-if="!capabilities && !error" role="status">正在读取原生推理能力…</p>
    <p v-if="capabilities?.image_ready === false" class="error" role="alert">原生任务镜像尚未就绪，暂不能创建任务。</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="capabilities?.tasks?.length" class="muted xs">冻结任务：{{ capabilities.tasks.map((task: any) => task.label).join(' · ') }}</p>

    <template v-if="!model">
      <div class="form-grid">
        <label for="ufo-model-name">模型名称</label><input id="ufo-model-name" v-model="sourceName" :disabled="busy" />
        <label for="ufo-model-path">本地来源目录（只读）</label><input id="ufo-model-path" v-model="sourcePath" :disabled="busy" />
      </div>
      <p class="mono xs">目标接口：{{ capabilities?.interface_id || '读取中…' }}</p>
      <button type="button" :disabled="busy || !capabilities || capabilities.image_ready === false || !sourceName.trim() || !sourcePath.trim()" @click="check('imports')">检查导入配置</button>
    </template>

    <template v-else>
      <div class="kpis">
        <div class="card"><div class="l">导入检查</div><div class="v xs">{{ currentModel?.card?.readiness || '已登记' }}</div></div>
        <div class="card"><div class="l">导出等价性</div><div class="v xs">{{ exported || status?.exports?.length ? '已导出' : '未导出' }}</div></div>
        <div class="card"><div class="l">MJLab 仿真</div><div class="v xs">{{ reportState('mjlab') }}</div></div>
        <div class="card"><div class="l">MuJoCo 迁移诊断</div><div class="v xs">{{ reportState('mujoco') }}</div></div>
        <div class="card"><div class="l">目标设备验证</div><div class="v xs">未验证</div></div>
        <div class="card"><div class="l">仿真交付包</div><div class="v xs">{{ status?.bundles?.length || 0 }}</div></div>
      </div>
      <template v-if="!exported">
        <p>先导出冻结的 ONNX、任务提示与推理配置，再验证该导出模型。</p>
        <button type="button" :disabled="busy || !capabilities || capabilities.image_ready === false" @click="check('export')">检查导出配置</button>
        <p v-for="child in status?.exports || []" :key="child.id"><router-link :to="`/models/${child.id}`">打开导出模型：{{ child.name || child.id }}</router-link></p>
      </template>
      <template v-else>
        <p><router-link :to="{ path: '/deploy/simulation', query: { model_id: currentModel.id } }">将此冻结模型部署到实时仿真</router-link></p>
        <div class="form-grid">
          <label for="ufo-eval-backend">仿真后端</label>
          <select id="ufo-eval-backend" v-model="backend" :disabled="busy"><option value="mjlab">MJLab / MuJoCo-Warp</option><option value="mujoco">MuJoCo CPU</option></select>
          <label for="ufo-eval-profile">评估规模</label>
          <select id="ufo-eval-profile" v-model="profile" :disabled="busy"><option value="smoke">Smoke · 三项任务各 1 case</option><option value="acceptance" disabled>Acceptance · 12 cases（尚未完成运行验证）</option></select>
        </div>
        <p class="muted xs">0.7 m/s 用于 smoke 诊断；任务判据、seed 和预算由后端冻结。两个后端同属 MuJoCo 系。</p>
        <p class="muted xs">当前已验证 smoke 完整流程。正式 acceptance 的完整运行时长与资格仍待验证。</p>
        <button type="button" :disabled="busy || !capabilities || capabilities.image_ready === false" @click="check('evals')">检查仿真配置</button>
        <p class="muted xs">两后端报告及必需媒体完整后可构建仿真交付包；任务失败会保留在报告中。</p>
        <button type="button" :disabled="busy || !capabilities || capabilities.image_ready === false" @click="check('bundles')">检查交付包证据</button>
      </template>
    </template>

    <div v-if="preview" class="native-preview">
      <p>已检查：{{ actionLabels[preview.action as Action] }} · <code>{{ preview.plan.hash }}</code></p>
      <details><summary>冻结计划与资源</summary><pre class="mono xs">{{ JSON.stringify({ plan: preview.plan, resources: preview.resources }, null, 2) }}</pre></details>
      <button class="primary" type="button" :disabled="busy || !preview.plan?.hash" @click="submit">{{ busy ? '提交中…' : `创建${actionLabels[preview.action as Action]}任务` }}</button>
    </div>
    <p v-if="busy && !preview" role="status">正在检查或提交任务…</p>

    <div v-if="jobs.length">
      <h4>运行记录</h4>
      <p class="muted xs">有活动任务时每 5 秒刷新，最多 10 分钟；之后可手动刷新。</p>
      <table><thead><tr><th>任务</th><th>状态</th><th>运行 / 错误</th></tr></thead><tbody>
        <tr v-for="job in jobs" :key="job.id"><td>{{ job.name || job.kind || job.id }}<br><span class="mono xs">{{ job.id }}</span></td><td>{{ job.status }}</td>
          <td><router-link v-if="runId(job)" :to="`/train/runs/${runId(job)}`">查看 Run {{ runId(job) }}</router-link><router-link v-else to="/train/queue">查看队列</router-link><p v-if="job.error" class="error">{{ job.error }}</p></td></tr>
      </tbody></table>
    </div>

    <div v-if="model && !loading && !reports.length" class="muted">尚无原生仿真报告。</div>
    <article v-for="report in reports" :key="report.id" class="native-report">
      <h4>{{ report.backend }} · {{ report.profile }} · {{ report.qualification || '未判定' }}</h4>
      <p class="mono xs">{{ report.id }} · <router-link v-if="report.run_id" :to="`/train/runs/${report.run_id}`">查看评估 Run</router-link></p>
      <p>成功率：{{ report.metrics?.success_rate == null ? '未提供' : `${(report.metrics.success_rate * 100).toFixed(1)}%` }}</p>
      <p v-if="report.metrics?.success_rate === 0" class="muted">本轮无成功案例。</p><p v-if="report.metrics?.success_rate === 1" class="muted">本轮无失败案例。</p>
      <div v-for="file in report.files || []" :key="file.path">
        <a :href="file.url" target="_blank" rel="noopener">{{ file.path }}</a>
        <img v-if="imageFile(file.path)" :src="file.url" :alt="`${report.backend} 仿真图表：${file.path}`" loading="lazy" @error="mediaErrors[file.url] = '图表加载失败'" />
        <p v-if="mediaErrors[file.url]" class="error" role="alert">{{ mediaErrors[file.url] }}：{{ file.path }}</p>
      </div>
      <p v-if="!report.metrics?.cases?.length" class="error">缺少逐 case 结果。</p>
      <details v-for="item in report.metrics?.cases || []" :key="item.case_id" class="native-case">
        <summary>{{ item.case_id }} · {{ item.task }} · seed {{ item.seed }} · {{ item.status }} · {{ item.steps }} steps</summary>
        <p>终止原因：{{ item.reason || '未提供' }}</p>
        <video v-if="item.video_url" :src="item.video_url" controls preload="metadata" :aria-label="`${item.case_id} 仿真录像`" @error="mediaErrors[item.video_url] = '录像加载失败'" />
        <p v-else class="error">未提供可播放录像，媒体证据不完整。</p>
        <p v-if="mediaErrors[item.video_url]" class="error" role="alert">{{ mediaErrors[item.video_url] }}</p>
        <pre v-if="item.metrics" class="mono xs">{{ JSON.stringify(item.metrics, null, 2) }}</pre>
      </details>
    </article>
    <div v-if="status?.bundles?.length">
      <h4>仿真交付包</h4>
      <p v-for="bundle in status.bundles" :key="bundle.id"><span class="mono xs">{{ bundle.id }}</span> · simulation_only · 未签名 · 目标硬件未验证 · <a v-if="bundle.download_url" :href="bundle.download_url" download>下载 .ebp</a></p>
      <router-link to="/deploy/bundles">查看交付包列表</router-link>
    </div>
  </section>
</template>

<style scoped>
.ufo-native-panel { margin-bottom: 1rem; }
.native-preview, .native-report { margin-top: 1rem; padding-top: .75rem; border-top: 1px solid var(--line); }
.native-case { margin: .6rem 0; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
video, img { display: block; width: 100%; max-width: 720px; margin: .6rem 0; }
</style>
