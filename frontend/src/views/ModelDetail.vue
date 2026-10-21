<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import DeleteButton from "../components/DeleteButton.vue";
import LineageGraph from "../components/LineageGraph.vue";
import UfoNativePanel from "../components/UfoNativePanel.vue";

const route = useRoute();
const router = useRouter();
const data = ref<any>(null);
const trainingSource = ref<any>(null);
const edges = ref<any[]>([]);
const lineageNodes = ref<any[]>([]);
const evalProfile = ref<"smoke" | "standard">("smoke");
const evalBackend = ref<"isaac" | "mujoco">("isaac");
const evalGoal = ref<"auto" | "stand_walk" | "stability" | "tracking">("auto");
const evalSeed = ref("");
const recordVideo = ref(false);
const evalPreview = ref<any>(null);
const evalBusy = ref(false);
const evalError = ref("");
const isRpo = computed(() => data.value?.card?.schema_version === "eb/rpo-wbc-model/v1");
const isNative = computed(() => ["eb/rpo-ufo-native-trained-model/v1", "eb/rpo-ufo-native-model/v1", "eb/rpo-ufo-native-export/v1"].includes(data.value?.card?.schema_version));
const trainingPage = computed(() => trainingSource.value?.train_page_id || trainingSource.value?.train_job_id);
const trainingName = computed(() => trainingSource.value?.train_job_name || trainingSource.value?.card?.train_job_id || trainingPage.value);
watch([evalBackend, evalGoal, evalProfile, evalSeed, recordVideo], () => { evalPreview.value = null; evalError.value = ""; });

function evalBody() {
  return { model_version_id: String(route.params.id), backend: evalBackend.value, profile: evalProfile.value,
    ...(evalGoal.value === "auto" ? {} : {suite_id:evalGoal.value === "tracking"
      ? (evalBackend.value === "isaac" ? "suite_rpo_v2_wbc_isaac_v1" : "suite_rpo_v2_wbc_sim2sim_v1")
      : `suite_rpo_v2_wbc_${evalBackend.value}_${evalGoal.value}_${evalGoal.value === 'stand_walk' ? 'v2' : 'v1'}`}),
    ...(String(evalSeed.value).trim() ? {seed:Number(evalSeed.value)} : {}),
    record_video: recordVideo.value };
}
async function previewRpo() {
  evalBusy.value = true; evalError.value = "";
  try { evalPreview.value = await api("/evals/rpo/preview", {method:"POST", body:JSON.stringify(evalBody())}); }
  catch (e: any) { evalError.value = e.message; }
  finally { evalBusy.value = false; }
}
async function startRpo() {
  if (!evalPreview.value) return;
  evalBusy.value = true; evalError.value = "";
  try {
    const job = await api("/evals/rpo/jobs", {method:"POST",
      headers:{"Idempotency-Key":globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`},
      body:JSON.stringify({...evalBody(), preview_hash:evalPreview.value.plan.hash})});
    await router.push(`/eval/jobs/${job.id}`);
  } catch (e: any) { evalError.value = e.message; }
  finally { evalBusy.value = false; }
}

watch(() => route.params.id, async id => {
  data.value = null;
  trainingSource.value = null;
  edges.value = [];
  lineageNodes.value = [];
  const model = await api(`/models/${id}`);
  const lineage = await api(`/models/${id}/lineage`);
  let source = model;
  const seen = new Set([model.id]);
  while (source.card?.training_provenance === "platform" && !source.train_page_id && !source.card?.train_job_id
      && source.card?.source_model_id && seen.size < 4 && !seen.has(source.card.source_model_id)) {
    seen.add(source.card.source_model_id);
    try { source = await api(`/models/${source.card.source_model_id}`); }
    catch { break; }
    if (route.params.id !== id) return;
  }
  if (route.params.id !== id) return;
  data.value = model;
  trainingSource.value = source;
  edges.value = lineage.edges || [];
  lineageNodes.value = lineage.nodes || [];
}, { immediate: true });
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">模型 / {{ route.params.id }}</p>
        <h2>{{ data?.name || route.params.id }} <span class="muted xs mono">{{ route.params.id }}</span></h2>
      </div>
      <router-link class="badge" to="/models">← 列表</router-link>
      <DeleteButton v-if="data" :path="`/models/${data.id}`" :name="data.name" :disabled="data.kind === 'downloading'" @deleted="router.push('/models')" />
    </div>
    <p class="muted" v-if="data">
      {{ isNative ? 'UFO/RPO 原生模型' : data.kind || data.provenance || "model" }} · {{ data.path || data.card?.package_path || data.card?.source_path || "—" }}
    </p>
    <div class="kpis" v-if="data">
      <div class="card"><div class="l">类型</div><div class="v">{{ data.kind || "version" }}</div></div>
      <div class="card"><div class="l">本体</div><div class="v">{{ data.embodiment_id || data.embodiment_env_id || "—" }}</div></div>
      <div class="card"><div class="l">训练 Job</div><div class="v xs"><router-link v-if="trainingPage" :to="`/train/${trainingPage}`">{{ trainingName }}</router-link><span v-else>{{ trainingName || '—' }}</span><br v-if="trainingPage"><span class="mono muted xs">{{ trainingPage }}</span></div></div>
      <div class="card"><div class="l">父模型</div><div class="v"><router-link v-if="data.parent_model_id || data.card?.source_model_id" :to="`/models/${data.parent_model_id || data.card.source_model_id}`">{{ data.parent_model_id || data.card.source_model_id }}</router-link><span v-else>—</span></div></div>
    </div>
    <UfoNativePanel v-if="isNative" :key="data.id" :model="data" />
    <div class="panel" v-if="data?.card">
      <h3>模型元数据</h3>
      <p>来源 Run：<router-link v-if="data.card.run_id" :to="`/train/runs/${data.card.run_id}`">{{ data.card.run_id }}</router-link><span v-else>—</span> · checkpoint step {{ data.card.step ?? '—' }}</p>
      <p>登记时间：{{ data.created_at || '—' }}</p>
      <p>状态：{{ data.card.readiness || '—' }} · 真机验证：{{ data.card.hardware_verified ? '是' : '否' }}</p>
      <p class="mono xs">产物：{{ data.artifact?.storage || data.card.checkpoint_path || data.card.package_path || '—' }}</p>
      <p class="mono xs">SHA256：{{ data.artifact?.sha256 || data.card.artifact_sha256 || '—' }} · {{ data.artifact?.size_bytes ?? '—' }} bytes</p>
      <p class="mono xs">本体环境：{{ data.embodiment_env_id || '—' }} · 输入 {{ data.io_signature?.obs ?? '—' }} / 输出 {{ data.io_signature?.act ?? '—' }} · {{ data.env?.policy_hz ?? '—' }} Hz</p>
      <details v-if="data.card.training"><summary>冻结训练配置</summary><pre class="mono xs">{{ JSON.stringify(data.card.training, null, 2) }}</pre></details>
    </div>
    <div class="panel" v-if="data?.card?.schema_version === 'eb/rpo-flat-model/v1'">
      <h3>官方 RPO-Flat 策略</h3>
      <p>Actor：780 → 512 → 256 → 128 → 23，ELU；Critic：1390 → 512 → 256 → 128 → 1。checkpoint 保存观测归一化统计，动作以 50 Hz 输出 23 个关节位置目标。</p>
      <p class="muted">训练与测试覆盖官方完整速度、横移和转向指令。标准评估为 100 个独立 20 秒案例，至少 80 个通过；训练完成后仍需独立评估。</p>
      <router-link :to="`/eval/new?model_id=${data.id}`">创建 RPO-Flat 评估</router-link>
    </div>
    <div class="panel" v-if="isRpo">
      <h3>策略结构与输出</h3>
      <div class="kpis">
        <div class="card"><div class="l">输入 · 10 帧 × 89</div><div class="v">890</div><div class="muted xs">角速度 3＋重力 3＋速度指令 3＋关节位置 23＋速度 23＋上次动作 23＋上肢参考 11</div></div>
        <div class="card"><div class="l">Actor · Tanh MLP</div><div class="v xs">890 → 256 → 256 → 256 → 23</div><div class="muted xs">评估导出确定性均值；训练时学习对角高斯方差</div></div>
        <div class="card"><div class="l">输出 · 50 Hz</div><div class="v">23 关节目标</div><div class="muted xs">默认关节角＋0.25 × 策略动作，经 PD 执行；不是电机力矩</div></div>
        <div class="card"><div class="l">训练专用 Critic</div><div class="v xs">1500 → 256 → 256 → 256 → 1</div><div class="muted xs">附加机体速度、足部接触和关节动力学；部署时不使用</div></div>
      </div>
      <p class="muted xs">训练阶段：{{ data.card.template_id === 'rpo_v2_wbc_stand_walk_ppo' ? '平地直立＋0.3m/s 直行，上肢中立' : '行走＋上肢随机参考跟踪' }}。模型结构相同，训练目标不同。</p>
    </div>
    <div class="panel" v-if="isRpo">
      <h3>RPO 闭环仿真评估</h3>
      <p class="muted">使用该模型的 checkpoint 导出确定性 ONNX，在 Isaac Lab 或 MuJoCo 中执行。录像模式在 headless 容器中离屏渲染，完成后可在报告页回放。</p>
      <label>仿真环境 <select v-model="evalBackend"><option value="isaac">Isaac Lab（训练同域）</option><option value="mujoco">MuJoCo（跨域）</option></select></label>
      <label>评估目标 <select v-model="evalGoal"><option value="auto">与训练阶段匹配</option><option value="stand_walk">直立＋0.3m/s 直行</option><option value="stability">身体稳定</option><option value="tracking">旧版上肢与速度跟踪</option></select></label>
      <p class="muted xs">直立＋直行：偶数 seed 原地站 20 秒、漂移 ≤ 0.5m；奇数 seed 直行 20 秒、前进 ≥ 4m。低于 0.5m 或倾斜超过 0.8rad 即跌倒并结束。</p>
      <label>评估规模 <select v-model="evalProfile"><option value="smoke">Smoke · 3 seeds</option><option value="standard">标准 · 12 seeds</option></select></label>
      <label>Seed（可选）<input v-model="evalSeed" type="number" min="0" max="1000000" step="1" placeholder="留空使用默认 seed" /></label>
      <p class="muted xs">留空按规模使用默认 seed；填写后仅运行该 seed，结果只用于诊断。</p>
      <label><input v-model="recordVideo" type="checkbox"> 录制视频</label>
      <button type="button" :disabled="evalBusy" @click="previewRpo">检查评估配置</button>
      <p v-if="evalPreview" class="mono xs">{{ evalPreview.plan.hash }} · {{ evalPreview.image }} · {{ evalPreview.plan.seeds.length }} seeds</p>
      <button v-if="evalPreview" type="button" :disabled="evalBusy" @click="startRpo">创建评估任务</button>
      <p v-if="evalError" class="error" role="alert">{{ evalError }}</p>
    </div>
    <div class="panel" v-if="data?.eval_reports?.length">
      <h3>评估记录</h3>
      <table><thead><tr><th>报告</th><th>结果</th><th>成功率</th><th>创建时间</th></tr></thead>
        <tbody><tr v-for="report in data.eval_reports" :key="report.id">
          <td><router-link :to="`/eval/reports/${report.id}`">{{ report.id }}</router-link></td>
          <td>{{ report.gate_results?.qualification || '—' }}</td>
          <td>{{ report.metrics?.success_rate == null ? '—' : `${(report.metrics.success_rate * 100).toFixed(1)}%` }}</td>
          <td>{{ report.created_at }}</td>
        </tr></tbody></table>
    </div>
    <div class="panel">
      <h3>DAG 血缘</h3>
      <p class="muted xs">本体 + 数据 + 基座 + 训练 → 产物</p>
      <LineageGraph :nodes="lineageNodes" :edges="edges" />
    </div>
  </div>
</template>
