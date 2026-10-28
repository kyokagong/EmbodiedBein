<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const route = useRoute();
const templates = ref<any[]>([]);
const environments = ref<any[]>([]);
const embodiments = ref<any[]>([]);
const datasets = ref<any[]>([]);
const weights = ref<any[]>([]);
const images = ref<any[]>([]);
const err = ref("");
const previewError = ref("");
const preview = ref<any>(null);
const submitting = ref(false);
const ready = ref(false);
const previewing = ref(false);
const form = ref({
  name: "", model_name: "", template_id: "", environment_id: "", embodiment_id: "", image: "", gpus: 2,
  gym_id: "", dataset_ids: [] as string[], base_model_id: "",
  max_epochs: "", max_steps: "", total_num_envs: "", eval_num_envs: "", rollout_epoch: "",
  micro_batch_size: "", global_batch_size: "", save_interval: "", val_check_interval: "", seed: "",
  ppo_max_epochs: "1",
  lr: "", value_lr: "", extra_text: "",
  native_target: 112, native_buffer: 262144, native_seed: 42,
});
const selectedTemplate = computed(() => templates.value.find(t => t.id === form.value.template_id));
const controlledPbfm = computed(() => form.value.template_id === "rpo_pbfm_fb_depth_native"
  ? selectedTemplate.value?.controlled_training : null);
const compatibleEnvironments = computed(() => environments.value.filter(e => (e.status === "active" ||
  (controlledPbfm.value && e.status === "draft")) && e.usage === "train"
  && e.template_id === form.value.template_id && e.embodiment_env_id === selectedTemplate.value?.embodiment_env_id));
const selectedEnvironment = computed(() => compatibleEnvironments.value.find(e => e.id === form.value.environment_id));
const templateDefaults = computed(() => Object.fromEntries((selectedTemplate.value?.default_overrides || [])
  .map((line: string) => line.split("=", 2))));
const defaultLabel = (key: string) => templateDefaults.value[key] ? `默认 ${templateDefaults.value[key]}` : "配置默认";
const isRpoWbc = computed(() => ["rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo"].includes(form.value.template_id));
const isRpoAmp = computed(() => form.value.template_id === "rpo_v2_amp_stand_walk_ppo");
const isRpoFlat = computed(() => form.value.template_id === "rpo_v2_flat_ppo");
const isUfoNative = computed(() => selectedTemplate.value?.kind === "native_rl");
const isPbfm = computed(() => form.value.template_id === "rpo_pbfm_fb_depth_native");
const compatibleEmbodiments = computed(() => embodiments.value.filter(e => !selectedTemplate.value?.embodiment_id || e.id === selectedTemplate.value.embodiment_id));
const compatibleDatasets = computed(() => {
  const environmentIds = new Set((embodiments.value.find(e => e.id === form.value.embodiment_id)?.envs || []).map((e: any) => e.id));
  return datasets.value.filter(d => d.usage !== "eval" && environmentIds.has(d.embodiment_env_id) &&
    (!selectedTemplate.value?.embodiment_env_id || d.embodiment_env_id === selectedTemplate.value.embodiment_env_id) &&
    (!isUfoNative.value || (d.source_kind === (isPbfm.value ? "pbfm_rpo_motion" : "ufo_native_motion") && d.status === "validated" &&
      d.validation?.scope === (isPbfm.value ? "pbfm_rpo_motion_v1" : "ufo_native_motion_v1") && d.validation?.training_ready === true)) &&
    (!isRpoAmp.value || (d.purpose === "motion_prior" && d.validation?.training_ready === true && d.status === "validated")));
});
const compatibleWeights = computed(() => weights.value.filter(w =>
  (!w.embodiment_id || w.embodiment_id === form.value.embodiment_id) &&
  (!selectedTemplate.value?.model_framework || w.framework === selectedTemplate.value.model_framework)));
const missingInput = computed(() => Boolean(!form.value.name.trim() || (!controlledPbfm.value && !form.value.model_name.trim()) || !form.value.embodiment_id || !selectedTemplate.value ||
  (selectedTemplate.value.ready === false && !controlledPbfm.value) || !selectedEnvironment.value ||
  (selectedTemplate.value.requires_dataset && !form.value.dataset_ids.some(Boolean)) ||
  (selectedTemplate.value.requires_base_model && !form.value.base_model_id)));
let controller: AbortController | undefined;
let timer: number | undefined;
let generation = 0;
let disposed = false;
let nativeSubmitKey = "";
onUnmounted(() => { disposed = true; generation++; clearTimeout(timer); controller?.abort(); });

onMounted(async () => {
  try {
    const [ts, es, ds, ws, ims, envs] = await Promise.all([
      api("/train/templates"), api("/embodiments"), api("/datasets"), api("/weights"), api("/envs/images"), api("/environments?usage=train"),
    ]);
    if (disposed) return;
    templates.value = ts.items || [];
    embodiments.value = es.items || [];
    datasets.value = ds.items || [];
    weights.value = (ws.items || []).filter((w: any) => !["downloading", "download_failed"].includes(w.kind));
    images.value = ims.items || [];
    environments.value = envs.items || [];
    const requested = environments.value.find(e => e.id === route.query.environment_id && e.status === "active");
    form.value.template_id = requested?.template_id || templates.value.find(t => t.id === route.query.template_id)?.id || templates.value[0]?.id || "";
    await nextTick();
    if (requested) form.value.environment_id = requested.id;
    ready.value = true;
  } catch (e: any) { if (!disposed) err.value = e.message; }
});

watch(() => form.value.template_id, () => {
  const t = selectedTemplate.value;
  form.value.embodiment_id = t?.embodiment_id || "";
  form.value.environment_id = compatibleEnvironments.value[0]?.id || "";
  form.value.dataset_ids = [];
  form.value.base_model_id = "";
  form.value.image = "";
  form.value.gym_id = "";
  if (isRpoAmp.value || isRpoFlat.value || isUfoNative.value) form.value.gpus = 1;
  if (controlledPbfm.value) {
    form.value.gpus = controlledPbfm.value.gpus;
    form.value.native_target = controlledPbfm.value.target_updates;
    form.value.native_buffer = controlledPbfm.value.buffer_size;
    form.value.native_seed = controlledPbfm.value.seed;
  }
  for (const [field, key] of Object.entries({
    max_epochs: "runner.max_epochs", max_steps: "runner.max_steps",
    total_num_envs: "env.train.total_num_envs", eval_num_envs: "env.eval.total_num_envs",
    rollout_epoch: "env.train.rollout_epoch", micro_batch_size: "actor.micro_batch_size",
    global_batch_size: "actor.global_batch_size", save_interval: "runner.save_interval",
    val_check_interval: "runner.val_check_interval", lr: "actor.optim.lr",
    value_lr: "actor.optim.value_lr", seed: "actor.seed",
  })) (form.value as any)[field] = templateDefaults.value[key] || "";
  form.value.extra_text = "";
});
watch(() => form.value.embodiment_id, () => {
  form.value.gym_id = "";
  form.value.dataset_ids = [];
  form.value.base_model_id = "";
});
watch(form, () => {
  nativeSubmitKey = "";
  generation++;
  controller?.abort();
  clearTimeout(timer);
  preview.value = null;
  previewError.value = "";
  previewing.value = false;
  if (ready.value && !missingInput.value) timer = window.setTimeout(refreshPreview, 300);
}, { deep: true });

function payload() {
  if (controlledPbfm.value) return {
    name: form.value.name.trim(), dataset_id: form.value.dataset_ids[0],
    environment_id: form.value.environment_id, mode: controlledPbfm.value.mode,
    gpus: controlledPbfm.value.gpus, num_envs_per_rank: controlledPbfm.value.num_envs_per_rank,
    target_updates: controlledPbfm.value.target_updates, timeout_seconds: controlledPbfm.value.timeout_seconds,
    ...(preview.value?.preview_hash ? { preview_hash: preview.value.preview_hash } : {}),
  };
  return {
    name: form.value.name.trim(), model_name: form.value.model_name.trim(), template_id: form.value.template_id,
    environment_id: form.value.environment_id,
    embodiment_id: form.value.embodiment_id, image: form.value.image || null,
    gpus: Number(form.value.gpus), gym_id: form.value.gym_id || null,
    dataset_ids: form.value.dataset_ids.filter(Boolean), base_model_id: form.value.base_model_id || null,
    ...Object.fromEntries(["max_epochs", "max_steps", "total_num_envs", "eval_num_envs", "rollout_epoch", "micro_batch_size", "global_batch_size", "save_interval", "val_check_interval", "seed"].map(k => [k, (form.value as any)[k] === "" ? null : Number((form.value as any)[k])])),
    lr: form.value.lr || null,
    value_lr: form.value.value_lr || null,
    extra_overrides: form.value.extra_text.split("\n").map(s => s.trim()).filter(Boolean),
    ...(isUfoNative.value ? { [isPbfm.value ? "pbfm_training" : "native_training"]: { target_updates: Number(form.value.native_target),
      buffer_size: Number(form.value.native_buffer), seed: Number(form.value.native_seed) },
      preview_hash: preview.value?.preview_hash || null } : {}),
  };
}
async function refreshPreview() {
  const current = generation;
  controller = new AbortController();
  previewing.value = true;
  try {
    const result = await api(controlledPbfm.value ? "/pbfm/runtime-probe/preview" : "/train/preview", { method: "POST", body: JSON.stringify(payload()), signal: controller.signal });
    if (current === generation) preview.value = result;
  } catch (e: any) {
    if (current === generation && e.name !== "AbortError") previewError.value = e.message;
  } finally { if (current === generation) previewing.value = false; }
}
async function createPipeline() {
  if (missingInput.value || submitting.value) return;
  submitting.value = true;
  try {
    await api("/train/libero-pipeline", { method: "POST", body: JSON.stringify({
      ...payload(), ppo_max_epochs: Number(form.value.ppo_max_epochs),
    }) });
    router.push("/train/pipelines");
  } catch (e: any) { err.value = e.message; }
  finally { submitting.value = false; }
}

async function submit() {
  if (missingInput.value || !preview.value || preview.value.errors?.length || previewing.value || submitting.value) return;
  submitting.value = true;
  err.value = "";
  try {
    if (isUfoNative.value && !nativeSubmitKey) nativeSubmitKey = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    const row = await api(controlledPbfm.value ? "/pbfm/runtime-probe" : "/train", { method: "POST", body: JSON.stringify(payload()),
      ...(isUfoNative.value ? { headers: { "Idempotency-Key": nativeSubmitKey } } : {}) });
    if (!disposed) await router.push(`/train/${row.job_id || row.id}`);
  } catch (e: any) { if (!disposed) err.value = e.message; }
  finally { submitting.value = false; }
}
</script>
<template>
  <div>
    <div class="page-head"><div><p class="crumb">训练 / 新建</p><h2>新建 RLinf 作业</h2></div><router-link class="badge" to="/train">← 返回列表</router-link></div>
    <p class="error" role="alert">{{ err }}</p>
    <div class="grid2">
      <form class="panel" @submit.prevent="submit">
        <fieldset :disabled="submitting || !ready" style="border:0;padding:0;margin:0;min-width:0">
          <h3>配置</h3>
          <label class="form-label" for="train-name">任务名称</label><input id="train-name" class="form-input" v-model="form.name" required />
          <template v-if="!controlledPbfm"><label class="form-label" for="train-model-name">产出模型名称</label><input id="train-model-name" class="form-input" v-model="form.model_name" maxlength="120" required placeholder="例如：RPO WBC 步态模型" /></template>
          <p v-else class="muted xs">完成后自动登记同名训练模型，可从结果页查看、导出并独立评估。</p>
          <label class="form-label" for="train-template">模板</label>
          <select id="train-template" class="form-select" v-model="form.template_id" required><option v-for="t in templates" :key="t.id" :value="t.id">{{ t.label || t.id }} · {{ t.kind }}{{ t.controlled_training ? ' · 四卡分段验证' : t.ready === false ? ' · 未就绪' : '' }}</option></select>
          <label class="form-label" for="train-environment">训练环境</label>
          <select id="train-environment" class="form-select" v-model="form.environment_id" required><option value="">请选择环境</option><option v-for="e in compatibleEnvironments" :key="e.id" :value="e.id">{{ e.name }} · {{ e.id }}</option></select>
          <p v-if="!compatibleEnvironments.length" class="error xs">此模板暂无已发布的兼容训练环境。</p>
          <p v-if="selectedEnvironment" class="muted xs">任务：{{ selectedEnvironment.definition.task }}<br>奖励：{{ selectedEnvironment.definition.reward }}<br>失败：{{ selectedEnvironment.definition.failure }}<br>成功：{{ selectedEnvironment.definition.success }}</p>
          <p v-if="selectedTemplate?.ready === false && !controlledPbfm" class="error" role="alert">{{ selectedTemplate.blocked_reason }}</p>
          <p v-if="controlledPbfm" class="muted">四卡分段验证：累计更新112 → 368 → 512。每段完成后先独立评估，再从结果页决定续训；512为本轮终点。普通长训模板尚未开放。</p>
          <p class="muted xs">{{ selectedTemplate?.kind === 'sft' ? 'SFT：使用数据集监督微调。' : selectedTemplate?.kind === 'ppo' ? '在线 PPO：通过仿真交互训练，可不挂数据集。' : selectedTemplate?.kind === 'eval' ? '评估：运行模板指定的仿真套件。' : '按模板执行作业。' }}</p>
          <div class="grid2">
            <div><label class="form-label" for="train-gpus">GPU 数</label><select id="train-gpus" class="form-select" v-model.number="form.gpus" :disabled="!!controlledPbfm"><option :value="1">1 卡</option><option :value="2">2 卡</option><option :value="4">4 卡</option></select></div>
            <div v-if="isRpoWbc || isRpoFlat || isUfoNative"><label class="form-label" for="train-rpo-image">镜像</label><input id="train-rpo-image" class="form-input" :value="selectedTemplate?.default_image" disabled /></div>
            <div v-else><label class="form-label" for="train-image">镜像</label><select id="train-image" class="form-select" v-model="form.image"><option value="">模板默认</option><option v-for="im in images" :key="im.id" :value="im.name">{{ im.name }}</option></select></div>
            <div><label class="form-label" for="train-emb">机器人 / 本体</label><select id="train-emb" class="form-select" v-model="form.embodiment_id" required><option value="">请选择兼容本体</option><option v-for="e in compatibleEmbodiments" :key="e.id" :value="e.id">{{ e.display_name || e.id }}</option></select></div>
            <div v-if="!isRpoWbc && !isRpoFlat"><label class="form-label" for="train-data">数据集（{{ selectedTemplate?.requires_dataset ? '必填' : '可选' }}）</label><select id="train-data" class="form-select" v-model="form.dataset_ids[0]" :required="selectedTemplate?.requires_dataset"><option value="">请选择</option><option v-for="d in compatibleDatasets" :key="d.id" :value="d.id">{{ d.name }} · {{ d.embodiment_env_id }}</option></select></div>
            <div v-if="!isRpoWbc && !isRpoAmp && !isRpoFlat && !isUfoNative"><label class="form-label" for="train-base">基座模型（{{ selectedTemplate?.requires_base_model ? '必填' : '可选' }}）</label><select id="train-base" class="form-select" v-model="form.base_model_id" :required="selectedTemplate?.requires_base_model"><option value="">请选择</option><option v-for="w in compatibleWeights" :key="w.id" :value="w.id">{{ w.name }}</option></select><p v-if="selectedTemplate?.requires_base_model && !compatibleWeights.length" class="error xs">没有兼容基座模型，请先到“模型”页面下载或登记。</p></div>
          </div>
          <p v-if="isRpoFlat" class="muted xs">官方 RPO-Flat：按完整速度、横移和转向指令训练，保留 20% 站立指令、地形混合及随机化，由 RLinf 执行 PPO。标准评估运行 100 个独立 20 秒案例，以至少 80 个通过为目标。</p>
          <p v-if="form.template_id === 'rpo_v2_wbc_stand_walk_ppo'" class="muted xs">第一阶段：平地直立或以 0.3 m/s 直行；上肢固定中立位，倒地立即终止。完成后用直立＋直行套件分别验证，不以训练回报代替通过率。</p>
          <details v-if="form.template_id === 'rpo_v2_wbc_stand_walk_ppo'">
            <summary>查看训练环境、奖励和终止条件</summary>
            <p class="muted xs">Isaac Lab 平地；每个 20 秒 episode 固定站立或 0.3 m/s 直行，各占约一半；无横移、转向、推搡与上肢随机目标。</p>
            <p class="muted xs">主要奖励权重：速度跟踪 +3、直立 +0.5、yaw 跟踪 +1、上肢中立位 +1；姿态误差 -1、提前终止 -200、站立关节偏差 -0.2。奖励按 0.02 秒统一缩放，并保留足部、能耗和关节限位约束。</p>
            <p class="muted xs">基座高度 &lt; 0.50 m、倾角 &gt; 0.80 rad 或关键躯干接触时立即终止；达到 1000 步正常截断。训练指标不能代替独立评估和录像。</p>
          </details>
          <p v-else-if="isRpoWbc" class="muted xs">RPO WBC 使用仿真在线 PPO 从随机初始化训练；状态来自 Isaac 环境，奖励由行走、稳定和上半身跟踪计算，不使用 Qwen/VLA 基座或离线数据集。</p>
          <p v-if="isRpoAmp" class="muted xs">RPO AMP 使用已验证的训练动作集，经 RLinf 联合更新 PPO 策略与运动判别器。首期单 GPU；训练完成后仍须独立评估直立、行走和自然度。</p>
          <p v-if="form.template_id === 'libero_pi05_sft'" class="muted xs">默认冻结视觉语言模型，仅微调动作专家（train_expert_only）。基座已完成上游 LIBERO SFT，本作业继续微调。</p>
          <h3>训练超参数</h3>
          <template v-if="isUfoNative">
            <p class="muted xs">{{ isPbfm ? 'P-BFM 深度地形 FB-CPR-Aux' : 'UFO 原生 FB-CPR-Aux' }}，每卡 {{ controlledPbfm?.num_envs_per_rank || 1024 }} 个环境。按累计 agent 更新数分段保存；独立快评后再恢复下一段。</p>
            <p v-if="isPbfm" class="muted xs">23 关节 · 策略 50 Hz · 深度 8×18×32 · 七类地形；课程从 0 开始，每 1920 万全局环境步扩大难度。模型是否会通过地形须由独立评估验证。</p>
            <div class="grid2">
              <div><label class="form-label" for="native-target">累计 agent 更新目标（16 的倍数）</label><input id="native-target" class="form-input" type="number" min="112" :max="isPbfm ? 10000 : 1000000" step="16" v-model.number="form.native_target" :disabled="!!controlledPbfm" /></div>
              <div><label class="form-label" for="native-buffer">每卡 Replay transition 容量</label><input id="native-buffer" class="form-input" type="number" min="32768" :max="isPbfm ? 5120000 : 1048576" step="1024" v-model.number="form.native_buffer" :disabled="!!controlledPbfm" /></div>
              <div><label class="form-label" for="native-seed">随机种子</label><input id="native-seed" class="form-input" type="number" min="0" max="1000000" v-model.number="form.native_seed" :disabled="!!controlledPbfm" /></div>
            </div>
            <p class="muted xs">原生状态与全 rank Replay 可恢复；环境、RNG 和 latent 缓存重新初始化。训练完成后仍须独立验证任务能力。</p>
          </template>
          <template v-else>
          <p class="muted xs">已填入模板覆盖值；空白表示使用训练配置中的默认值。实际生效值见右侧预览。</p>
          <div class="grid2">
            <div><label class="form-label" for="train-epochs">max_epochs{{ selectedTemplate?.kind === 'ppo' ? '（训练迭代次数）' : '' }}</label><input id="train-epochs" class="form-input" type="number" min="1" step="1" v-model="form.max_epochs" :placeholder="defaultLabel('runner.max_epochs')" /></div>
            <div v-if="selectedTemplate?.kind === 'sft'"><label class="form-label" for="train-steps">max_steps</label><input id="train-steps" class="form-input" type="number" min="1" step="1" v-model="form.max_steps" :placeholder="defaultLabel('runner.max_steps')" /></div>
            <div v-if="selectedTemplate?.kind === 'ppo'"><label class="form-label" for="train-envs">训练环境数</label><input id="train-envs" class="form-input" type="number" min="1" step="1" v-model="form.total_num_envs" :placeholder="defaultLabel('env.train.total_num_envs')" /></div>
            <div v-if="selectedTemplate?.kind === 'ppo'"><label class="form-label" for="train-eval-envs">评估环境数</label><input id="train-eval-envs" class="form-input" type="number" min="1" step="1" v-model="form.eval_num_envs" :placeholder="defaultLabel('env.eval.total_num_envs')" /></div>
            <template v-if="selectedTemplate?.kind === 'sft' || selectedTemplate?.kind === 'ppo'">
              <div><label class="form-label" for="train-micro">micro_batch_size</label><input id="train-micro" class="form-input" type="number" min="1" step="1" v-model="form.micro_batch_size" :placeholder="defaultLabel('actor.micro_batch_size')" /></div>
              <div><label class="form-label" for="train-global">global_batch_size</label><input id="train-global" class="form-input" type="number" min="1" step="1" v-model="form.global_batch_size" :placeholder="defaultLabel('actor.global_batch_size')" /></div>
            </template>
            <div><label class="form-label" for="train-save">保存间隔</label><input id="train-save" class="form-input" type="number" min="-1" step="1" v-model="form.save_interval" :placeholder="defaultLabel('runner.save_interval')" /></div>
            <div v-if="selectedTemplate?.kind === 'ppo'"><label class="form-label" for="train-val">验证间隔</label><input id="train-val" class="form-input" type="number" min="-1" step="1" v-model="form.val_check_interval" :placeholder="defaultLabel('runner.val_check_interval')" /></div>
            <div v-if="selectedTemplate?.kind === 'ppo'"><label class="form-label" for="train-rollout">每次采样轮数</label><input id="train-rollout" class="form-input" type="number" min="1" step="1" v-model="form.rollout_epoch" :placeholder="defaultLabel('env.train.rollout_epoch')" /></div>
            <div><label class="form-label" for="train-seed">随机种子</label><input id="train-seed" class="form-input" type="number" min="0" step="1" v-model="form.seed" :placeholder="defaultLabel('actor.seed')" /></div>
          </div>
          <p v-if="selectedTemplate?.kind === 'ppo'" class="muted xs">环境数控制并行采样量；global batch 必须能被 micro batch × GPU 数整除。</p>
          <div class="grid2">
            <div><label class="form-label" for="train-lr">策略学习率</label><input id="train-lr" class="form-input" v-model="form.lr" :placeholder="defaultLabel('actor.optim.lr')" /></div>
            <div v-if="selectedTemplate?.kind === 'ppo'"><label class="form-label" for="train-value-lr">价值学习率</label><input id="train-value-lr" class="form-input" v-model="form.value_lr" :placeholder="defaultLabel('actor.optim.value_lr')" /></div>
          </div>
          <label class="form-label" for="train-extra">高级 Hydra Overrides（一行一个）</label><textarea id="train-extra" class="form-textarea" v-model="form.extra_text" placeholder="runner.save_interval=20" />
          <p class="muted xs">高级覆盖与上方字段同名时，以高级覆盖为准。</p>
          </template>
          <template v-if="form.template_id === 'libero_pi05_sft'">
            <label class="form-label" for="pipeline-ppo-epochs">管线后续 PPO max_epochs</label>
            <input id="pipeline-ppo-epochs" class="form-input" type="number" min="1" max="10000" step="1" v-model="form.ppo_max_epochs" />
            <p class="muted xs">仅用于管线的 PPO 阶段。训练完成后，到评估菜单选择 checkpoint 和 LIBERO 套件创建独立任务。</p>
            <button type="button" :disabled="missingInput || !preview || previewing || !Number.isInteger(Number(form.ppo_max_epochs)) || Number(form.ppo_max_epochs) < 1 || Number(form.ppo_max_epochs) > 10000" @click="createPipeline">创建 SFT → spatial PPO 管线</button>
          </template>
          <button class="primary" style="margin-top:1rem;width:100%" :disabled="missingInput || !preview || previewing || !!preview?.errors?.length">{{ submitting ? '提交中…' : '提交作业' }}</button>
        </fieldset>
      </form>
      <div class="panel">
        <h3>已解析配置与输入挂载</h3>
        <p class="muted xs">环境 {{ selectedEnvironment?.name || '—' }} · 本体接口 {{ selectedTemplate?.embodiment_env_id || '—' }} · 任务 {{ selectedTemplate?.task_id || '—' }}</p>
        <p v-if="preview" class="muted xs">产出模型：{{ controlledPbfm ? form.name.trim() : preview.model_name }}</p>
        <p class="error" role="alert">{{ previewError }}</p>
        <p v-if="previewing" role="status">正在校验输入并生成预览…</p>
        <pre class="preview-box" v-else-if="preview">{{ JSON.stringify(preview, null, 2) }}</pre>
        <p v-else class="muted">填写任务名、模型名及模板必需的输入后显示实际 argv、Overrides 与挂载。</p>
      </div>
    </div>
  </div>
</template>
