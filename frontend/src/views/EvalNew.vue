<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";

const route = useRoute(); const router = useRouter();
const suites = ref<any[]>([]); const candidates = ref<any[]>([]); const checkpoints = ref<any[]>([]); const rpoCandidates = ref<any[]>([]);
const environments = ref<any[]>([]); const environmentId = ref("");
const runtimeProfiles = ref<any[]>([]);
const benchmark = ref<"rpo" | "robotwin" | "libero">("rpo");
const rpoBackend = ref<"isaac" | "mujoco">("isaac");
const name = ref(""); const source = ref(""); const suiteId = ref("");
const profile = ref("smoke"); const runtimeProfileId = ref(""); const resourceVariantId = ref("");
const rpoSeed = ref("");
const recordVideo = ref(false);
const preview = ref<any>(null); const error = ref(""); const busy = ref(false);
const key = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
const visibleSuites = computed(() => suites.value.filter(s => benchmark.value === "rpo"
  ? ["rpo_wbc", "rpo_amp", "rpo_flat", "rpo_bfm"].includes(s.spec?.benchmark) && s.spec?.backend === rpoBackend.value
    && !s.id.endsWith("_stand_walk_v1") : benchmark.value === "robotwin"
  ? s.spec?.benchmark === "robotwin" : s.spec?.suite?.startsWith("libero_"))
  .sort((a, b) => Number(b.spec?.criterion === "stand_walk") - Number(a.spec?.criterion === "stand_walk")
    || Number(b.spec?.criterion === "stability") - Number(a.spec?.criterion === "stability")));
const compatibleEnvironments = computed(() => environments.value.filter(e => e.status === "active" && e.usage === "eval" && e.suite_id === suiteId.value));
const selectedEnvironment = computed(() => compatibleEnvironments.value.find(e => e.id === environmentId.value));
const selected = computed(() => benchmark.value === "rpo" ? rpoCandidates.value.find(c => c.id === source.value)
  : benchmark.value === "robotwin" ? candidates.value.find(c => c.id === source.value)
  : checkpoints.value.find(c => c.key === source.value));
const isUfo = computed(() => selected.value?.execution?.provider_id === "ufo");
const compatibleRpoCandidates = computed(() => rpoCandidates.value.filter(c =>
  c.embodiment_env_id === suites.value.find(s => s.id === suiteId.value)?.embodiment_env_id));
const runtimeProfile = computed(() => runtimeProfiles.value.find(p => p.id === runtimeProfileId.value));
const runtimeVariants = computed(() => runtimeProfile.value?.variants || []);
const resourceVariant = computed(() => runtimeVariants.value.find((v: any) => v.id === resourceVariantId.value));
const canPreview = computed(() => (benchmark.value === "rpo" || !!name.value.trim()) && !!suiteId.value && !!selected.value && !!selectedEnvironment.value
  && (benchmark.value !== "robotwin" || (selected.value.readiness !== "blocked"
    && runtimeProfile.value?.status === "ready" && !!resourceVariant.value)));

async function loadCandidates() {
  source.value = ""; candidates.value = [];
  if (benchmark.value === "robotwin" && suiteId.value) {
    const data = await api(`/evals/candidates?suite_id=${encodeURIComponent(suiteId.value)}`);
    candidates.value = data.items || [];
  }
  source.value = benchmark.value === "rpo" ? compatibleRpoCandidates.value[0]?.id || ""
    : benchmark.value === "robotwin" ? candidates.value[0]?.id || "" : checkpoints.value[0]?.key || "";
}
watch(benchmark, async () => {
  suiteId.value = visibleSuites.value[0]?.id || "";
  profile.value = "smoke"; recordVideo.value = false; preview.value = null; error.value = "";
  try { await loadCandidates(); } catch (e: any) { error.value = e.message; }
});
watch(suiteId, async () => {
  environmentId.value = compatibleEnvironments.value[0]?.id || "";
  const role = suites.value.find(s => s.id === suiteId.value)?.spec?.evaluation_role;
  if (["validation", "acceptance"].includes(role)) { profile.value = role; recordVideo.value = true; }
  preview.value = null; error.value = "";
  try { await loadCandidates(); } catch (e: any) { error.value = e.message; }
});
watch(rpoBackend, async () => {
  suiteId.value = visibleSuites.value[0]?.id || ""; preview.value = null; error.value = "";
  try { await loadCandidates(); } catch (e: any) { error.value = e.message; }
});
watch(runtimeProfileId, () => {
  resourceVariantId.value = runtimeProfile.value?.recommended_variant_id || runtimeVariants.value[0]?.id || "";
});
watch([name, source, profile, rpoSeed, recordVideo, runtimeProfileId, resourceVariantId, environmentId], () => { preview.value = null; error.value = ""; });
onMounted(async () => {
  try {
    const [s, c, r, rpo, envs] = await Promise.all([
      api("/evals/suites"), api("/evals/checkpoints"), api("/evals/runtime-profiles"), api("/evals/rpo/candidates"), api("/environments?usage=eval")]);
    suites.value = s.items || []; checkpoints.value = c.items || []; runtimeProfiles.value = r.items || []; rpoCandidates.value = rpo.items || [];
    environments.value = envs.items || [];
    runtimeProfileId.value = runtimeProfiles.value.find(p => p.status === "ready")?.id || "";
    resourceVariantId.value = runtimeProfile.value?.recommended_variant_id || "";
    if (!visibleSuites.value.length) benchmark.value = "robotwin";
    suiteId.value = visibleSuites.value[0]?.id || "";
    await loadCandidates();
    const match = checkpoints.value.find(v => v.run_id === route.query.run_id && String(v.step) === String(route.query.step));
    if (match) { benchmark.value = "libero"; suiteId.value = visibleSuites.value[0]?.id || ""; source.value = match.key; }
    const rpoMatch = rpoCandidates.value.find(v => v.id === route.query.model_id);
    if (rpoMatch) {
      benchmark.value = "rpo"; rpoBackend.value = "isaac";
      await nextTick();
      suiteId.value = visibleSuites.value.find(s => s.embodiment_env_id === rpoMatch.embodiment_env_id)?.id || "";
      await nextTick();
      source.value = rpoMatch.id;
    }
    const requested = environments.value.find(e => e.id === route.query.environment_id && e.status === "active");
    if (requested) {
      const suite = suites.value.find(s => s.id === requested.suite_id);
      benchmark.value = ["rpo_wbc", "rpo_amp", "rpo_flat", "rpo_bfm"].includes(suite?.spec?.benchmark) ? "rpo" : suite?.spec?.benchmark === "robotwin" ? "robotwin" : "libero";
      if (benchmark.value === "rpo") rpoBackend.value = suite?.spec?.backend === "mujoco" ? "mujoco" : "isaac";
      await nextTick();
      suiteId.value = requested.suite_id;
      await nextTick();
      environmentId.value = requested.id;
    }
  } catch (e: any) { error.value = e.message; }
});
function body() {
  if (benchmark.value === "rpo") return { model_version_id:selected.value?.id,
    backend:rpoBackend.value, suite_id:suiteId.value, environment_id:environmentId.value,
    profile: isUfo.value ? profile.value : profile.value === "standard" ? "standard" : "smoke",
    ...(String(rpoSeed.value).trim() ? {seed:Number(rpoSeed.value)} : {}),
    record_video: isUfo.value && ["validation", "acceptance"].includes(profile.value) ? true : recordVideo.value };
  if (benchmark.value === "robotwin") return { name:name.value.trim(), suite_id:suiteId.value, environment_id:environmentId.value,
    profile:profile.value, model_version_id:selected.value?.id,
    runtime_profile_id:runtimeProfileId.value, resource_variant_id:resourceVariantId.value };
  return { name:name.value.trim(), suite_id:suiteId.value, environment_id:environmentId.value, profile:profile.value,
    checkpoint_run_id:selected.value?.run_id, checkpoint_step:selected.value?.step };
}
async function check() {
  if (!canPreview.value) return;
  busy.value = true; error.value = "";
  try { preview.value = await api(benchmark.value === "rpo" ? "/evals/rpo/preview" : "/evals/previews", { method:"POST", body:JSON.stringify(body()) }); }
  catch (e: any) { error.value = e.message; } finally { busy.value = false; }
}
async function submit() {
  if (!preview.value || busy.value) return;
  busy.value = true; error.value = "";
  try {
    const row = await api(benchmark.value === "rpo" ? "/evals/rpo/jobs" : "/evals/jobs", { method:"POST", headers:{"Idempotency-Key":key},
      body:JSON.stringify({...body(), preview_hash:preview.value.plan.hash}) });
    await router.push(`/eval/jobs/${row.id}`);
  } catch (e: any) { error.value = e.message; } finally { busy.value = false; }
}
</script>
<template>
  <div>
    <div class="page-head"><div><p class="crumb"><router-link to="/eval">评估任务</router-link> / 创建</p><h2>创建闭环仿真评估</h2></div></div>
    <p class="muted">平台冻结模型版本、套件、case、seed 与镜像，由 RLInf 驱动仿真并回收逐案例证据。</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <div class="panel"><h3>① 输入</h3>
      <label v-if="benchmark !== 'rpo'">名称 <input v-model="name" maxlength="120" placeholder="例如 pi05-robotwin-smoke" /></label>
      <label>基准 <select v-model="benchmark"><option value="rpo">RPO V2</option><option value="robotwin">RoboTwin</option><option value="libero">LIBERO</option></select></label>
      <label v-if="benchmark === 'rpo'">仿真环境 <select v-model="rpoBackend"><option value="isaac">Isaac Lab（训练同域）</option><option value="mujoco">MuJoCo（跨域）</option></select></label>
      <label v-if="benchmark === 'rpo'">模型 / checkpoint <select v-model="source"><option disabled value="">选择兼容 RPO RL 模型</option><option v-for="c in compatibleRpoCandidates" :key="c.id" :value="c.id">{{ c.name }} · step {{ c.step }} · {{ c.readiness }}</option></select></label>
      <label v-if="benchmark === 'robotwin'">模型版本 <select v-model="source"><option disabled value="">选择不可变模型版本</option><option v-for="c in candidates" :key="c.id" :value="c.id">{{ c.name }} · {{ c.readiness }}</option></select></label>
      <label v-else-if="benchmark === 'libero'">checkpoint <select v-model="source"><option disabled value="">选择已纳管训练权重</option><option v-for="c in checkpoints" :key="c.key" :value="c.key">{{ c.name }} / step {{ c.step }}</option></select></label>
      <div v-if="selected?.blockers?.length" class="error"><p v-for="b in selected.blockers" :key="b.code">{{ b.code }} · {{ b.message }}</p></div>
      <p v-if="selected?.artifact_sha256" class="muted xs">{{ selected.artifact_sha256 }} · {{ selected.io_signature?.observation_profile }} → {{ selected.io_signature?.action_profile }}</p>
    </div>
    <div class="panel"><h3>② 冻结套件</h3>
      <label>套件 <select v-model="suiteId"><option v-for="s in visibleSuites" :key="s.id" :value="s.id">{{ s.name }}{{ benchmark === 'rpo' && !['stability', 'stand_walk'].includes(s.spec?.criterion) ? ' · 跟踪精度' : '' }}</option></select></label>
      <label>评估环境 <select v-model="environmentId"><option disabled value="">选择兼容环境</option><option v-for="e in compatibleEnvironments" :key="e.id" :value="e.id">{{ e.name }} · {{ e.id }}</option></select></label>
      <p v-if="suiteId && !compatibleEnvironments.length" class="error xs">此套件暂无已发布的兼容评估环境。</p>
      <p v-if="selectedEnvironment" class="muted xs">任务：{{ selectedEnvironment.definition.task }}<br>奖励：{{ selectedEnvironment.definition.reward }}<br>失败：{{ selectedEnvironment.definition.failure }}<br>成功：{{ selectedEnvironment.definition.success }}</p>
      <p v-if="benchmark === 'rpo' && suiteId" class="muted xs">{{ visibleSuites.find(s => s.id === suiteId)?.spec?.criterion === 'stability'
        ? '身体稳定：达到 1000 步且未跌倒、无数值异常即为案例成功；低于 0.5m 或倾角超过 0.8rad 即结束。跟踪误差仅用于诊断。'
        : visibleSuites.find(s => s.id === suiteId)?.spec?.criterion === 'stand_walk'
        ? '偶数 seed 原地站立 20 秒、漂移 ≤ 0.5m；奇数 seed 以 0.3m/s 直行 20 秒、前进 ≥ 4m；全程保持直立。'
        : visibleSuites.find(s => s.id === suiteId)?.spec?.criterion === 'velocity_commands'
        ? '官方速度与转向：每例 20 秒，按官方范围采样并重采样指令；无跌倒、NaN 或硬越限，平移速度向量 RMSE ≤ 0.20m/s、转向 RMSE ≤ 0.30rad/s。标准 100 例至少 80 例通过。'
        : '跟踪精度：还需满足速度与上肢跟踪误差阈值。' }}</p>
      <p v-if="isUfo" class="muted xs">UFO / RLinf · 源联合更新 {{ selected.step }}；周期快评与最终验收分别冻结 seed，必需录像须完整。</p>
      <label>规模 <select v-model="profile"><option value="smoke">冒烟</option><option value="validation" v-if="benchmark === 'robotwin' || isUfo">验证</option><option value="standard">标准</option><option value="acceptance" v-if="benchmark === 'robotwin' || isUfo">验收</option></select></label>
      <label v-if="benchmark === 'rpo'">Seed（可选）<input v-model="rpoSeed" type="number" min="0" max="1000000" step="1" placeholder="留空使用默认 seed" /></label>
      <p v-if="benchmark === 'rpo'" class="muted xs">留空按规模使用默认 seed；填写后仅运行该 seed，结果只用于诊断。</p>
      <label v-if="benchmark === 'rpo'"><input v-model="recordVideo" type="checkbox" /> 录制评估视频（完成后可在评估报告中播放）</label>
      <fieldset v-if="benchmark === 'robotwin'"><legend>运行档位</legend>
        <label v-for="item in runtimeProfiles" :key="item.id">
          <input v-model="runtimeProfileId" type="radio" :value="item.id" :disabled="item.status !== 'ready'" />
          {{ item.name }} · {{ item.description }} · {{ item.status === 'ready' ? '可提交' : '未通过 probe' }}
          <span v-if="item.blockers?.length" class="error">（{{ item.blockers[0].message }}）</span>
        </label>
      </fieldset>
      <details v-if="benchmark === 'robotwin'"><summary>高级配置</summary>
        <label>已验证资源变体 <select v-model="resourceVariantId">
          <option v-for="item in runtimeVariants" :key="item.id" :value="item.id">
            {{ item.gpus }} GPU · {{ item.env_worker_count }} Worker / {{ item.parallel_envs }} env · 容器内存 {{ item.memory_gb }} GiB · 共享内存 {{ item.shm_gb }} GiB
          </option>
        </select></label>
        <p v-if="resourceVariant" class="muted">模型副本 {{ resourceVariant.policy_worker_count }} · 推理 micro-batch {{ resourceVariant.inference_micro_batch_size }} · {{ runtimeProfile?.scheduling_mode }}</p>
      </details>
      <p v-if="!visibleSuites.length" class="error">固定源码、seed 或资产尚未验证，暂无可提交套件。</p>
      <p v-else class="muted xs">{{ visibleSuites.find(s => s.id === suiteId)?.spec?.protocol || 'fixed' }} · {{ visibleSuites.find(s => s.id === suiteId)?.seeds?.length || 0 }} cases</p>
    </div>
    <div class="panel"><h3>③ 冻结预览</h3><button type="button" :disabled="!canPreview || busy" @click="check">{{ busy ? '检查中…' : '检查输入与计划' }}</button>
      <div v-if="preview"><p>计划 {{ benchmark === 'rpo' ? preview.plan.seeds.length : preview.plan.planned }} 条 · qualification：not_requested</p><p class="mono xs">{{ preview.plan.hash }}</p>
        <p>镜像 {{ preview.image }} · GPU {{ preview.resources.gpus }}<span v-if="benchmark === 'robotwin'"> 卡 · 容器内存 {{ preview.resources.memory_gb }} GiB · 共享内存 {{ preview.resources.shm_gb }} GiB · 模型副本 {{ preview.plan.model.runtime.policy_worker_count }} · 总并发环境 {{ preview.plan.model.runtime.parallel_envs }} · 推理 micro-batch {{ preview.plan.model.runtime.inference_micro_batch_size }}</span></p>
        <p v-if="benchmark === 'rpo'">仿真环境：{{ preview.plan.backend === 'isaac' ? 'Isaac Lab' : 'MuJoCo' }} · 成功标准：{{ preview.plan.criterion === 'stand_walk' ? '直立＋直行' : preview.plan.criterion === 'stability' ? '存活 1000 步' : '跟踪精度' }} · 评估视频：{{ preview.plan.record_video ? '录制' : '不录制' }}</p>
        <p v-if="benchmark === 'robotwin'">Headless：是 · 离屏三相机：是 · 视频关闭：是</p>
        <details><summary>解析后的命令与只读挂载</summary><pre class="mono xs">{{ JSON.stringify({ argv: preview.argv, mounts: preview.mounts }, null, 2) }}</pre></details>
        <button type="button" :disabled="busy" @click="submit">创建评估任务</button></div>
    </div>
  </div>
</template>
