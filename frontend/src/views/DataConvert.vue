<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "../api/client";

const step = ref(1);
const datasets = ref<any[]>([]);
const srcId = ref("");
const mapNote = ref("动词表 / 关节重定向：沿用源仓脚本，不在 UI 改已发布接口数字");
const preview = ref<any>(null);
const funnel = ref<any>(null);
const out = ref<any>(null);
const err = ref("");

onMounted(async () => {
  datasets.value = (await api("/datasets")).items || [];
  srcId.value = datasets.value[0]?.id || "";
});

const src = computed(() => datasets.value.find((d) => d.id === srcId.value));

async function loadPreview() {
  if (!srcId.value) return;
  preview.value = await api(`/datasets/${srcId.value}/slice/preview`, { method: "POST", body: JSON.stringify({}) });
  const vid = src.value?.versions?.[0]?.id || (await api(`/datasets/${srcId.value}`)).versions?.[0]?.id;
  const ds = await api(`/datasets/${srcId.value}`);
  const v = ds.versions?.[0]?.id;
  if (v) funnel.value = await api(`/datasets/${srcId.value}/versions/${v}/funnel`);
}

async function submit(kind: string) {
  err.value = "";
  try {
    out.value = await api("/jobs", {
      method: "POST",
      body: JSON.stringify({
        spec: {
          kind,
          embodiment_env_id: src.value?.embodiment_env_id || "ee_bhl_fixedhand_10dof_v1",
          params: { src: src.value?.meta?.root || "/data/kk/berkeley_ego_data", dataset_id: srcId.value },
        },
      }),
    });
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <h2>转换向导</h2>
    <p class="muted">五步：源 → 映射 → 预览 5 条 → 干跑漏斗 → 提交 Job。镜像不存在会 infra_failed，不生成空轨迹。</p>
    <div class="steps">
      <span :class="{ on: step === 1 }">1 源</span>
      <span :class="{ on: step === 2 }">2 映射</span>
      <span :class="{ on: step === 3 }">3 预览</span>
      <span :class="{ on: step === 4 }">4 干跑漏斗</span>
      <span :class="{ on: step === 5 }">5 提交</span>
    </div>
    <div class="panel" v-if="step === 1">
      <h3>已登记 DatasetVersion</h3>
      <select v-model="srcId">
        <option v-for="d in datasets" :key="d.id" :value="d.id">{{ d.name }} ({{ d.id }})</option>
      </select>
      <p class="muted xs">许可传染：{{ src?.license_gate?.label || src?.license }}。NC 在第 1 步就写进目标草稿。</p>
      <button type="button" @click="step = 2">下一步</button>
    </div>
    <div class="panel" v-if="step === 2">
      <h3>映射</h3>
      <p>{{ mapNote }}</p>
      <p class="muted xs">G2 预检：当前仓无机器人动作维，retarget 之后才能与 fixedhand 本体环境兼容。</p>
      <div class="row">
        <button type="button" @click="step = 1">上一步</button>
        <button type="button" @click="step = 3; loadPreview()">下一步</button>
      </div>
    </div>
    <div class="panel" v-if="step === 3">
      <h3>预览样本（失败/扫描样本优先）</h3>
      <ul class="xs">
        <li v-for="(s, i) in (preview?.preview || []).slice(0, 5)" :key="i">{{ s.path || s }}</li>
      </ul>
      <p class="muted" v-if="!preview?.preview?.length">无样本可预览。</p>
      <div class="row">
        <button type="button" @click="step = 2">上一步</button>
        <button type="button" @click="step = 4">下一步</button>
      </div>
    </div>
    <div class="panel" v-if="step === 4">
      <h3>干跑漏斗（当前扫描数字，不是 155k）</h3>
      <p v-if="funnel">source {{ funnel.stages?.[0]?.n }} → valid_episode {{ funnel.stages?.[2]?.n }} → retargeted {{ funnel.stages?.[3]?.n }}</p>
      <p class="muted xs">retargeted=0 表示尚未重定向。预计留存以扫描为准。</p>
      <div class="row">
        <button type="button" @click="step = 3">上一步</button>
        <button type="button" @click="step = 5">下一步</button>
      </div>
    </div>
    <div class="panel" v-if="step === 5">
      <h3>提交 Job</h3>
      <div class="row">
        <button type="button" @click="submit('data.convert')">data.convert</button>
        <button type="button" @click="submit('data.retarget')">data.retarget</button>
        <button type="button" @click="submit('data.compile')">data.compile</button>
      </div>
      <p class="error">{{ err }}</p>
      <pre v-if="out">{{ JSON.stringify(out, null, 2) }}</pre>
      <button type="button" @click="step = 4">上一步</button>
    </div>
  </div>
</template>
