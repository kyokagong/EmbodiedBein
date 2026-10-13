<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import MotionBrowser from "../components/MotionBrowser.vue";

const route = useRoute();
const data = ref<any>(null);
const tree = ref<any>(null);
const dir = ref("");
const media = ref("");
const err = ref("");
const segment = ref<any>(null);
const episodes = ref<any>({ items: [], total: 0, next_cursor: null });
const cursor = ref(0);
let refreshTimer: ReturnType<typeof setInterval> | undefined;
const isMotion = computed(() => ['rpo_bfm_motion', 'rpo_ufo_robot_state'].includes(data.value?.format));

async function setUsage(usage: "train" | "eval") {
  try {
    data.value = await api(`/datasets/${route.params.id}/usage`, {
      method: "PATCH", body: JSON.stringify({ usage }),
    });
    await load();
  } catch (e: any) { err.value = e.message; }
}

async function loadEpisodes(offset = 0) {
  episodes.value = await api(`/datasets/${route.params.id}/episodes?cursor=${offset}&limit=50`);
  cursor.value = offset;
}

async function validate(reindex = false) {
  try { await api(`/datasets/${route.params.id}/validate?reindex=${reindex}`, { method: "POST" }); await load(); }
  catch (e: any) { err.value = e.message; }
}
function playEpisode(stream: any) {
  segment.value = stream;
  media.value = `/api/v1/datasets/${route.params.id}/media?path=${encodeURIComponent(stream.path)}#t=${stream.from_timestamp},${stream.to_timestamp}`;
}
function constrain(event: Event) {
  const video = event.target as HTMLVideoElement;
  if (!segment.value) return;
  if (video.currentTime < segment.value.from_timestamp) video.currentTime = segment.value.from_timestamp;
  if (video.currentTime >= segment.value.to_timestamp) { video.pause(); video.currentTime = segment.value.from_timestamp; }
}

onMounted(async () => {
  await load();
  refreshTimer = setInterval(async () => {
    if (["libero_benchmark", "robotwin"].includes(data.value?.source_kind) && ["queued", "admitted", "running"].includes(data.value.status)) {
      try {
        data.value = await api(`/datasets/${route.params.id}`);
        if (data.value.status === "validated") await loadTree("");
      } catch (e: any) { err.value = e.message; }
    }
  }, 5000);
});
onUnmounted(() => { if (refreshTimer) clearInterval(refreshTimer); });

async function load() {
  try {
    err.value = "";
    data.value = await api(`/datasets/${route.params.id}`);
    media.value = "";
    if (data.value.status === "validated" && !isMotion.value) await loadTree("");
    if (data.value.source_kind !== "libero_benchmark" && !isMotion.value) await loadEpisodes();
  } catch (e: any) { err.value = e.message; }
}
watch(() => route.params.id, load);

async function loadTree(path: string) {
  dir.value = path;
  try {
    tree.value = await api(`/datasets/${route.params.id}/tree?path=${encodeURIComponent(path)}&page=1&page_size=80`);
    err.value = "";
  } catch (e: any) {
    tree.value = null;
    err.value = e.message;
  }
}

function open(item: any) {
  if (item.is_dir) {
    loadTree(item.path);
    return;
  }
  const lower = item.name.toLowerCase();
  if (lower.endsWith(".mp4") || lower.endsWith(".webm") || lower.endsWith(".mkv")) {
    segment.value = null;
    media.value = `/api/v1/datasets/${route.params.id}/media?path=${encodeURIComponent(item.path)}`;
  }
}

function up() {
  if (!dir.value) return;
  const parts = dir.value.split("/").filter(Boolean);
  parts.pop();
  loadTree(parts.join("/"));
}

function gb(n: number | null) {
  if (!n) return "—";
  return (n / 1e9).toFixed(2) + " GB";
}
</script>
<template>
  <div v-if="data">
    <div class="page-head">
      <div>
        <p class="crumb">数据 / {{ data.id }}</p>
        <h2>{{ data.name }} <span class="badge">{{ data.status || "ready" }}</span></h2>
      </div>
      <router-link class="badge" to="/data">← 列表</router-link>
    </div>
    <dl class="dl">
      <dt>来源</dt><dd>{{ data.source_kind || "path" }} · {{ data.source_uri || "—" }}</dd>
      <dt>本体</dt><dd>{{ data.embodiment_name || "未关联" }} · <span class="mono">{{ data.embodiment_id || "—" }}</span></dd>
      <dt>本体环境</dt><dd class="mono">{{ data.embodiment_env_id || "—" }}</dd>
      <dt>用途</dt><dd v-if="data.source_kind === 'libero_benchmark'">评估测试场景</dd><dd v-else><select :value="data.usage || 'train'" @change="setUsage(($event.target as HTMLSelectElement).value as 'train' | 'eval')"><option value="train">训练集</option><option value="eval">评估集</option></select></dd>
      <dt>落地</dt><dd>{{ data.dest_path || data.meta?.root || "—" }}</dd>
      <dt>体积</dt><dd>{{ gb(data.total_bytes || data.overview?.bytes) }}</dd>
      <template v-if="data.source_kind === 'robotwin'">
        <dt>原生本体</dt><dd>{{ data.meta?.native_embodiment || '—' }}</dd>
        <dt>场景</dt><dd>{{ data.meta?.scenario || '—' }}</dd>
        <dt>适配方式</dt><dd>{{ data.meta?.compatibility || '—' }}</dd>
      </template>
    </dl>
    <p v-if="data.source_kind === 'libero_benchmark'">基准校验：{{ data.validation?.status || "not_validated" }} · {{ data.summary_stats?.benchmark?.suites ? '4 个套件，每套件 10 个任务' : '等待任务与初始状态校验' }}</p>
    <p v-else>结构校验：{{ data.validation?.status || "not_validated" }} · training_ready={{ !!data.validation?.training_ready }}</p>
    <p v-if="data.source_kind === 'libero_benchmark'" class="mono xs">{{ data.summary_stats?.benchmark?.sha256 || '' }}</p>
    <button @click="validate(data.source_kind === 'libero_benchmark' && data.status !== 'registered')" :disabled="!['registered', 'failed', 'invalid'].includes(data.status)">{{ data.source_kind === 'libero_benchmark' ? '下载 / 重试校验' : '提交下载 / 结构校验' }}</button>
    <button v-if="!['libero_benchmark', 'robotwin'].includes(data.source_kind)" @click="validate(true)" :disabled="!['validated', 'invalid', 'failed'].includes(data.status)">本地重新索引</button>
    <button @click="load">刷新状态</button>
    <MotionBrowser v-if="isMotion" :key="data.id" :dataset-id="data.id" :embodiment-id="data.embodiment_id || ''" />
    <p v-if="data.source_kind !== 'libero_benchmark' && !isMotion">{{ data.overview?.episodes }} episodes · {{ data.overview?.frames }} 帧 · {{ data.overview?.fps }} Hz · action {{ data.overview?.action_dim }} / state {{ data.overview?.state_dim }}</p>
    <p v-if="data.source_kind !== 'libero_benchmark' && !isMotion">G2：{{ data.overview?.g2 }} · {{ data.overview?.g2_foot }}</p>
    <p class="error xs">{{ data.error_message || err }}</p>
    <p v-if="data.source_kind !== 'libero_benchmark' && !isMotion">{{ episodes.total }} episodes
      <button :disabled="cursor === 0" @click="loadEpisodes(Math.max(0, cursor - 50))">上一页</button>
      <button :disabled="episodes.next_cursor === null" @click="loadEpisodes(episodes.next_cursor)">下一页</button>
    </p>
    <div v-for="ep in data.source_kind === 'libero_benchmark' || isMotion ? [] : episodes.items" :key="ep.episode_id">
      Episode {{ ep.episode_id }} · {{ ep.length }} 帧 · {{ ep.tasks?.join('; ') || '任务未索引' }}
      <button v-for="stream in ep.streams" :key="stream.key" @click="playEpisode(stream)">{{ stream.key }} · {{ stream.from_timestamp }}–{{ stream.to_timestamp }}s</button>
    </div>
    <div v-if="data.status === 'validated' && !isMotion" class="grid2">
      <div v-if="data.source_kind !== 'libero_benchmark'" class="panel">
        <h3>惰性文件树</h3>
        <div class="row" style="margin-bottom:.4rem">
          <button type="button" :disabled="!dir" @click="up">上级</button>
          <span class="mono xs muted">{{ dir || "/" }}</span>
        </div>
        <div v-if="tree">
          <div class="tree-row" v-for="it in tree.items" :key="it.path" @click="open(it)">
            <span>{{ it.is_dir ? "📁" : "📄" }}</span>
            <span>{{ it.name }}</span>
            <span class="muted xs" v-if="it.bytes">{{ it.bytes }} B</span>
          </div>
          <p class="muted xs" v-if="!tree.items?.length">空目录</p>
        </div>
        <p v-else class="muted xs">无法列出目录。确认 dest_path 在白名单内且可读。</p>
      </div>
      <div class="panel">
        <h3>媒体预览</h3>
        <video v-if="media" :src="media" @loadedmetadata="constrain" @timeupdate="constrain" @seeking="constrain" controls style="width:100%;border-radius:8px;background:#000" />
        <p v-else class="muted xs">点开 mp4 / webm 在此播放（走 HTTP Range）。</p>
      </div>
    </div>
  </div>
  <p v-else-if="err" class="error" role="alert">{{ err }}</p>
  <p v-else role="status">正在加载数据集…</p>
</template>
