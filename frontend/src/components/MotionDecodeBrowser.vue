<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";
import EbTable from "./EbTable.vue";

const props = defineProps<{ dataset: any }>();
const emit = defineEmits<{ refresh: [] }>();
const router = useRouter();
const derived = computed(() => props.dataset.meta?.operation === "motiondecode_valid_view_v1");
const creating = ref(false);
const error = ref("");
const query = ref("");
const episodes = ref<any>({ items: [], total: 0, next_cursor: null });
const cursor = ref(0);
const loading = ref(false);
const listError = ref("");
const selected = ref<any>(null);
const startFrame = ref(0);
const job = ref<any>(null);
const preview = ref<any>(null);
const previewError = ref("");
const previewLoading = ref(false);
const table = ref<any>(null);
const tablePage = ref(0);
const tableLoading = ref(false);
const tableError = ref("");
let listRequest = 0;
let previewRequest = 0;
let tableRequest = 0;
let viewRequest = 0;
let pollTimer: ReturnType<typeof setTimeout> | undefined;

function stopPreview() {
  ++previewRequest;
  ++tableRequest;
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = undefined;
  previewLoading.value = false;
  tableLoading.value = false;
}
function reset() {
  ++listRequest;
  ++viewRequest;
  stopPreview();
  creating.value = false;
  loading.value = false;
  selected.value = null;
  job.value = null;
  preview.value = null;
  table.value = null;
  query.value = "";
  episodes.value = { items: [], total: 0, next_cursor: null };
  cursor.value = 0;
  error.value = listError.value = previewError.value = tableError.value = "";
}
async function openValidView() {
  const token = ++viewRequest;
  creating.value = true;
  error.value = "";
  try {
    const child = props.dataset.meta?.valid_view_id
      ? { id: props.dataset.meta.valid_view_id }
      : await api(`/datasets/${props.dataset.id}/motiondecode/valid-view`, { method: "POST", body: "{}" });
    if (token === viewRequest) await router.push(`/data/${child.id}`);
  } catch (e: any) { if (token === viewRequest) error.value = e.message; }
  finally { if (token === viewRequest) creating.value = false; }
}
async function retryValidation() {
  const parent = props.dataset.meta?.parent_dataset_id;
  if (!parent) return;
  const token = ++viewRequest;
  creating.value = true;
  error.value = "";
  try {
    await api(`/datasets/${parent}/motiondecode/valid-view`, { method: "POST", body: "{}" });
    if (token === viewRequest) emit("refresh");
  } catch (e: any) { if (token === viewRequest) error.value = e.message; }
  finally { if (token === viewRequest) creating.value = false; }
}
async function loadEpisodes(offset = 0) {
  const token = ++listRequest;
  loading.value = true;
  listError.value = "";
  try {
    const result = await api(`/datasets/${props.dataset.id}/episodes?cursor=${offset}&limit=50&q=${encodeURIComponent(query.value.trim())}`);
    if (token !== listRequest) return;
    episodes.value = result;
    cursor.value = offset;
  } catch (e: any) { if (token === listRequest) listError.value = e.message; }
  finally { if (token === listRequest) loading.value = false; }
}
async function loadTable(page = 0) {
  if (!preview.value || page < 0 || page >= preview.value.table_pages) return;
  const token = ++tableRequest;
  const base = preview.value.table_base_url.replace(/^\/api\/v1/, "");
  tableLoading.value = true;
  tablePage.value = page;
  tableError.value = "";
  table.value = null;
  try {
    const result = await api(`${base}/${String(page).padStart(6, "0")}.json`);
    if (token !== tableRequest) return;
    table.value = result;
  } catch (e: any) { if (token === tableRequest) tableError.value = e.message; }
  finally { if (token === tableRequest) tableLoading.value = false; }
}
async function acceptJob(result: any, token: number, attempts: number) {
  if (token !== previewRequest) return;
  job.value = result;
  if (result.status === "succeeded") {
    if (!result.preview) throw new Error("预览任务未返回视频和数据表。");
    preview.value = result.preview;
    previewLoading.value = false;
    await loadTable(0);
  } else if (["failed", "infra_failed", "timeout", "blocked", "cancelled", "canceled"].includes(result.status)) {
    throw new Error(result.error || `预览任务 ${result.status}`);
  } else if (attempts >= 300) {
    throw new Error("等待预览超时，请重试以查询或生成预览。");
  } else {
    pollTimer = setTimeout(() => pollJob(result.id, token, attempts + 1), 2000);
  }
}
async function pollJob(id: string, token: number, attempts: number) {
  if (token !== previewRequest) return;
  try {
    const result = await api(`/datasets/${props.dataset.id}/motiondecode/previews/${id}`);
    await acceptJob(result, token, attempts);
  } catch (e: any) {
    if (token === previewRequest) { previewError.value = e.message; previewLoading.value = false; }
  }
}
async function generate(ep = selected.value, start = startFrame.value) {
  if (!ep) return;
  stopPreview();
  const token = previewRequest;
  selected.value = ep;
  startFrame.value = start;
  preview.value = table.value = job.value = null;
  previewError.value = tableError.value = "";
  if (!Number.isInteger(start) || start < 0 || start >= ep.length) {
    previewError.value = "起始帧须为动作范围内的整数（从 0 开始）。";
    return;
  }
  previewLoading.value = true;
  try {
    const result = await api(`/datasets/${props.dataset.id}/motiondecode/previews`, {
      method: "POST", body: JSON.stringify({ path: ep.path, start_frame: start }),
    });
    await acceptJob(result, token, 0);
  } catch (e: any) {
    if (token === previewRequest) { previewError.value = e.message; previewLoading.value = false; }
  }
}
watch(() => props.dataset.id, () => {
  reset();
  if (derived.value) void loadEpisodes();
}, { immediate: true });
watch(() => props.dataset.status, (status, previous) => {
  if (derived.value && status === "validated" && previous !== "validated") void loadEpisodes(0);
});
onUnmounted(reset);
</script>

<template>
  <section class="panel motiondecode" aria-label="MotionDecode 数据浏览">
    <h3>{{ derived ? '有效动作视图' : '保留原始数据，派生有效动作视图' }}</h3>
    <template v-if="!derived">
      <p>原始 CSV 保持不变；派生视图只排除已确认的空文件 Summer_Wind.csv，保留排除原因与来源哈希。</p>
      <button :disabled="creating" @click="openValidView">{{ creating ? '正在打开…' : '创建 / 打开有效视图' }}</button>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </template>
    <template v-else>
      <p>preview_ready={{ !!dataset.validation?.preview_ready }} · training_ready={{ !!dataset.validation?.training_ready }}。原始数据保留在父版本。</p>
      <router-link v-if="dataset.meta?.parent_dataset_id" :to="`/data/${dataset.meta.parent_dataset_id}`">查看原始父版本与排除记录</router-link>
      <button v-if="['failed', 'invalid'].includes(dataset.status) && dataset.meta?.parent_dataset_id" :disabled="creating" @click="retryValidation">{{ creating ? '正在提交…' : '重试有效版本校验' }}</button>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <p v-for="item in dataset.summary_stats?.exclusions || []" :key="item.path" class="xs exclusion">
        排除：{{ item.path }} · {{ item.reason }}<br><span class="mono">SHA256 {{ item.sha256 }}</span>
      </p>
      <form class="row" @submit.prevent="loadEpisodes(0)">
        <label>搜索动作 <input v-model="query" type="search" placeholder="例如 walking"></label>
        <button type="submit" :disabled="loading">搜索</button>
      </form>
      <p v-if="loading" role="status">正在加载动作…</p>
      <p v-if="listError" class="error" role="alert">{{ listError }} <button @click="loadEpisodes(cursor)">重试列表</button></p>
      <div class="scroll-table">
        <EbTable :columns="[{ key: 'path', label: '动作文件' }, { key: 'length', label: '源帧数' }, { key: 'actions', label: '预览' }]" :rows="episodes.items">
          <template #path="{ row }"><span class="mono xs">{{ row.path }}</span></template>
          <template #actions="{ row }"><button @click="generate(row, 0)">查看数据表 / 生成视频</button></template>
        </EbTable>
      </div>
      <div class="row" aria-label="动作分页">
        <span>{{ episodes.total }} 个动作 · 第 {{ Math.floor(cursor / 50) + 1 }} 页</span>
        <button :disabled="loading || cursor === 0" @click="loadEpisodes(Math.max(0, cursor - 50))">上一页</button>
        <button :disabled="loading || episodes.next_cursor == null" @click="loadEpisodes(episodes.next_cursor)">下一页</button>
      </div>
      <template v-if="selected">
        <h3 class="selected-path">{{ selected.path }}</h3>
        <p class="muted xs">真实 CSV 的 G1 运动学姿态回放；源文件不含时间戳，采用来源声明的 120 Hz。视频为 30 fps，每段最多 7200 源帧（60 秒）。不代表动力学、训练兼容性或验收结果。</p>
        <form class="row" @submit.prevent="generate()">
          <label>起始源帧（从 0 开始） <input v-model.number="startFrame" type="number" min="0" :max="selected.length - 1" step="1" required></label>
          <button type="submit" :disabled="previewLoading">生成此段 / 重试</button>
        </form>
        <p v-if="previewLoading" role="status">正在生成姿态回放… {{ job?.status || '提交中' }}</p>
        <p v-if="previewError" class="error" role="alert">{{ previewError }}</p>
        <template v-if="preview">
          <p>源帧 {{ preview.start_frame }}–{{ preview.end_frame }} / {{ preview.frames }} · {{ preview.source_fps }} Hz → {{ preview.fps }} fps · {{ preview.video_frames }} 视频帧</p>
          <video :key="preview.video_url" :src="preview.video_url" controls preload="metadata" aria-label="G1 姿态回放视频" @error="previewError = '视频无法播放，请重试生成或检查媒体访问。'" />
          <div class="row" aria-label="视频分段">
            <button :disabled="previewLoading || preview.start_frame === 0" @click="generate(selected, Math.max(0, preview.start_frame - 7200))">上一段</button>
            <button :disabled="previewLoading || preview.end_frame >= preview.frames" @click="generate(selected, preview.end_frame)">下一段</button>
          </div>
          <h4>原始数值表 · 全部源帧分页</h4>
          <p v-if="tableLoading" role="status">正在读取数据表…</p>
          <p v-if="tableError" class="error" role="alert">{{ tableError }} <button @click="loadTable(tablePage)">重试数据表</button></p>
          <div v-if="table" class="scroll-table" tabindex="0" role="region" aria-label="CSV 原始数值，横向滚动查看全部列">
            <table>
              <thead><tr><th scope="col">源帧</th><th v-for="column in table.columns" :key="column" scope="col">{{ column }}</th></tr></thead>
              <tbody><tr v-for="(row, index) in table.rows" :key="table.frame_start + index"><th scope="row">{{ table.frame_start + index }}</th><td v-for="(value, col) in row" :key="col">{{ value }}</td></tr></tbody>
            </table>
          </div>
          <div class="row" aria-label="源数据表分页">
            <span>第 {{ tablePage + 1 }} / {{ preview.table_pages }} 页 · 每页 {{ preview.table_page_size }} 源帧</span>
            <button :disabled="tableLoading || tablePage === 0" @click="loadTable(tablePage - 1)">上一页数据</button>
            <button :disabled="tableLoading || tablePage + 1 >= preview.table_pages" @click="loadTable(tablePage + 1)">下一页数据</button>
          </div>
        </template>
      </template>
    </template>
  </section>
</template>

<style scoped>
.motiondecode { margin-top: 1rem; min-width: 0; }
.scroll-table { overflow-x: auto; max-width: 100%; }
.scroll-table table { white-space: nowrap; }
.scroll-table th, .scroll-table td { padding: .4rem .6rem; }
.selected-path, .exclusion { overflow-wrap: anywhere; }
.row { flex-wrap: wrap; margin: .6rem 0; }
input { max-width: 100%; }
video { width: 100%; max-height: 520px; background: #111; border-radius: 8px; }
</style>
