<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api } from "../api/client";
import { frameAtTime, motionPose } from "../utils/motionPlayback";
import UrdfStage from "./UrdfStage.vue";
import Chart from "./Chart.vue";

const props = defineProps<{ datasetId: string; embodimentId: string }>();
const catalog = ref<any>(null);
const preview = ref<any>(null);
const trajectory = ref<any>(null);
const selected = ref("");
const query = ref("");
const error = ref("");
const loading = ref(false);
const submitting = ref(false);
const frame = ref(0);
const playing = ref(false);
const speed = ref(1);
const joint = ref(0);
let timer: ReturnType<typeof setInterval> | undefined;
let raf = 0, lastTime = 0, playbackTime = 0, request = 0;
let disposed = false;
const busy = computed(() => ["queued", "admitted", "preparing", "running", "collecting"].includes(catalog.value?.preview?.status));
const clips = computed(() => (catalog.value?.items || []).filter((c: any) =>
  `${c.id} ${c.source || ''} ${c.source_group || ''} ${c.split || ''}`.toLowerCase().includes(query.value.toLowerCase())));
const pose = computed(() => trajectory.value ? motionPose(trajectory.value, frame.value) : undefined);
const points = computed(() => (trajectory.value?.timestamps || []).map((t: number, i: number) =>
  ({ step: t, value: trajectory.value.q[i][joint.value] })));

async function loadCatalog() {
  try {
    catalog.value = await api(`/datasets/${props.datasetId}/motions`);
    if (!selected.value && catalog.value.items.length) selected.value = catalog.value.items[0].id;
    if (catalog.value.preview.status === "succeeded" && !trajectory.value && !loading.value) await selectClip(selected.value);
  } catch (e: any) { error.value = e.message; }
}

async function generate() {
  submitting.value = true;
  error.value = "";
  try {
    await api(`/datasets/${props.datasetId}/motion-preview?rebuild=${catalog.value?.preview.status === 'succeeded'}`, { method: "POST" });
    pause();
    trajectory.value = null;
    await loadCatalog();
  } catch (e: any) { error.value = e.message; }
  finally { submitting.value = false; }
}

function pause() { playing.value = false; cancelAnimationFrame(raf); }
async function selectClip(id: string) {
  pause();
  selected.value = id;
  trajectory.value = null;
  frame.value = 0;
  error.value = "";
  if (catalog.value?.preview.status !== "succeeded") return;
  loading.value = true;
  const token = ++request;
  try {
    const data = await api(`/datasets/${props.datasetId}/motions/${encodeURIComponent(id)}`);
    if (token === request && !disposed) trajectory.value = data;
  } catch (e: any) { if (token === request && !disposed) error.value = e.message; }
  finally { if (token === request && !disposed) loading.value = false; }
}

function tick(now: number) {
  if (!playing.value || !trajectory.value) return;
  playbackTime += (now - lastTime) / 1000 * speed.value;
  lastTime = now;
  const times = trajectory.value.timestamps;
  frame.value = frameAtTime(times, playbackTime);
  if (playbackTime >= times[times.length - 1]) { pause(); return; }
  raf = requestAnimationFrame(tick);
}
function play() {
  if (playing.value) { pause(); return; }
  if (!trajectory.value) return;
  if (frame.value === trajectory.value.timestamps.length - 1) frame.value = 0;
  playbackTime = trajectory.value.timestamps[frame.value];
  lastTime = performance.now();
  playing.value = true;
  raf = requestAnimationFrame(tick);
}
function seek(event: Event) { pause(); frame.value = Number((event.target as HTMLInputElement).value); }

onMounted(async () => {
  await loadCatalog();
  if (props.embodimentId) {
    try { preview.value = await api(`/embodiments/${props.embodimentId}/preview`); }
    catch (e: any) { error.value = `机器人模型不可用：${e.message}`; }
  }
  if (!disposed) timer = setInterval(() => { if (busy.value) void loadCatalog(); }, 2000);
});
onUnmounted(() => { disposed = true; request++; pause(); if (timer) clearInterval(timer); });
</script>

<template>
  <section class="panel">
    <h3>参考动作</h3>
    <p class="muted xs">回放保存的根位姿和关节角；属于参考轨迹预览，不代表策略物理运行或评估通过。距离 m，关节角 rad，时间 s。</p>
    <p v-if="!catalog && !error" role="status">正在加载动作索引…</p>
    <template v-if="catalog">
      <p>{{ catalog.total }} 个片段 · {{ catalog.frames }} 帧 · {{ catalog.seconds.toFixed(2) }} s · 23 关节</p>
      <div class="row">
        <button :disabled="busy || submitting" @click="generate">{{ busy ? '正在生成回放…' : catalog.preview.status === 'succeeded' ? '重新生成回放' : '生成动作回放' }}</button>
        <button @click="loadCatalog">刷新动作索引</button>
        <router-link v-if="catalog.preview.job_id" :to="catalog.preview.run_id ? `/train/runs/${catalog.preview.run_id}` : '/train/queue'">预览任务 · {{ catalog.preview.status }}</router-link>
      </div>
      <p v-if="catalog.preview.error" class="error">回放生成失败：{{ catalog.preview.error }}</p>
      <label>查找片段、来源组或划分 <input v-model="query" type="search" placeholder="例如 walk、train、dance1" /></label>
      <div class="motion-table">
        <table>
          <thead><tr><th>动作片段</th><th>来源组 / 划分</th><th>时长 / 频率</th><th>有效帧</th><th>查看</th></tr></thead>
          <tbody><tr v-for="c in clips" :key="c.id" :class="{ selected: selected === c.id }">
            <td class="mono xs">{{ c.id }}</td>
            <td>{{ c.source_group || '未知' }}<br><span class="badge">{{ c.split || '未划分' }}</span></td>
            <td>{{ (c.frames / c.fps).toFixed(2) }} s · {{ c.fps }} Hz</td>
            <td>{{ c.valid_frames ?? '未知' }} / {{ c.frames }}</td>
            <td><button @click="selectClip(c.id)" :aria-pressed="selected === c.id">查看动作</button></td>
          </tr></tbody>
        </table>
        <p v-if="!clips.length">没有匹配的动作片段。</p>
      </div>
      <p v-if="catalog.preview.status !== 'succeeded' && selected" class="muted">已选择 {{ selected }}。生成回放后可查看三维动作及关节曲线。</p>
      <p v-if="loading" role="status">正在加载 {{ selected }}…</p>
      <div v-if="trajectory" class="grid2">
        <div>
          <h4>{{ selected }} · 三维回放</h4>
          <UrdfStage v-if="preview?.urdf_url" :key="selected" :embodiment-id="embodimentId" :preview="preview" :motion-pose="pose" />
          <p v-else class="muted">机器人模型不可用，可在右侧查看关节曲线和数值。</p>
          <div class="row">
            <button @click="play">{{ playing ? '暂停' : '播放' }}</button>
            <label>速度 <select v-model.number="speed"><option :value=".25">0.25×</option><option :value=".5">0.5×</option><option :value="1">1×</option><option :value="2">2×</option></select></label>
            <span>{{ frame + 1 }} / {{ trajectory.timestamps.length }} 帧 · {{ trajectory.timestamps[frame].toFixed(3) }} s</span>
          </div>
          <label>时间轴 <input type="range" :value="frame" :max="trajectory.timestamps.length - 1" min="0" step="1" @input="seek" style="width:100%" /></label>
          <p :class="trajectory.valid[frame] ? 'muted' : 'warn'">{{ trajectory.valid[frame] ? '本帧通过参考数据筛查' : '本帧未通过参考数据筛查；按原始姿态回放' }}</p>
        </div>
        <div>
          <h4>关节角曲线</h4>
          <label>关节 <select v-model.number="joint"><option v-for="(name, i) in trajectory.joint_order" :key="name" :value="i">{{ name }}</option></select></label>
          <Chart :points="points" :playhead="trajectory.timestamps[frame]" title="时间 (s) / 关节角 (rad)" />
          <p>{{ trajectory.joint_order[joint] }}：{{ trajectory.q[frame][joint].toFixed(4) }} rad</p>
          <p>根位置：{{ trajectory.root_pos[frame].map((v: number) => v.toFixed(3)).join(', ') }} m</p>
          <details><summary>片段质量与来源</summary><p>{{ trajectory.clip.source_license || '来源许可未标注' }}</p><pre class="xs">{{ JSON.stringify(trajectory.clip.reference_quality || {}, null, 2) }}</pre><p class="mono xs">SHA256 {{ trajectory.clip_sha256 }}</p></details>
        </div>
      </div>
      <details><summary>数据来源与冻结版本</summary><p class="mono xs">索引 SHA256 {{ catalog.manifest_sha256 }}</p><div v-for="s in catalog.source_refs" :key="s.sha256"><p class="xs">{{ s.uri || s.url }} · {{ s.license || '许可未标注' }}</p><p class="mono xs">{{ s.sha256 }}</p></div></details>
    </template>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </section>
</template>

<style scoped>
.motion-table { max-height: 24rem; overflow: auto; margin: 1rem 0; }
.selected { background: rgba(83, 212, 255, .08); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
