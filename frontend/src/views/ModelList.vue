<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
import DeleteButton from "../components/DeleteButton.vue";
import { useListView } from "../composables/useListView";

const items = ref<any[]>([]);
const embodiments = ref<any[]>([]);
const robotwinModels = ref<any[]>([]);
const sources = ref<any[]>([]);
const show = ref(false);
const name = ref("");
const path = ref("");
const embodimentId = ref("");
const err = ref("");
const revision = ref("");
const acquiring = ref(false);
const repoId = ref("RLinf/RLinf-Pi05-LIBERO-SFT");
const selectedSource = () => sources.value.find(s => s.repo_id === repoId.value);
const sortOptions = [
  { key: "created", label: "创建时间", value: (w: any) => w.created_at },
  { key: "name", label: "名称", value: (w: any) => w.name },
  { key: "kind", label: "类型", value: (w: any) => w.kind },
  { key: "size", label: "体积", value: (w: any) => Number(w.size_bytes || 0) },
];
const filters = [
  { key: "name", label: "名称 / ID", type: "text" as const, value: (w: any) => `${w.name || ""} ${w.id || ""}` },
  { key: "kind", label: "类型", type: "text" as const, value: (w: any) => w.kind },
  { key: "path", label: "路径 / 下载状态", type: "text" as const, value: (w: any) => w.path || w.metrics_at_checkpoint?.download?.status },
  { key: "train", label: "训练任务", type: "text" as const, value: (w: any) => `${w.train_job_name || ""} ${w.train_job_id || ""}` },
];
const list = useListView(items, sortOptions, (w) => [w.name, w.id, w.kind, w.path, w.train_job_name, w.train_job_id, w.metrics_at_checkpoint?.download?.status].join(" "), filters, "created", "desc");

onMounted(load);

async function load() {
  const [weights, models] = await Promise.all([api("/weights"), api("/models")]);
  const modelNames = new Map((models.items || []).map((w: any) => [w.id, w.train_job_name]));
  items.value = [
    ...(models.versions || []).map((v: any) => ({...v, kind: v.tags?.includes("rpo-v2") ? "RPO WBC · RL" : "训练模型",
      path: v.card?.checkpoint_path || v.card?.package_path, train_job_id: v.train_page_id || v.card?.train_job_id,
      step: v.card?.step})),
    ...(weights.items || []).map((w: any) => ({ ...w, train_job_name: modelNames.get(w.id) })),
  ];
  embodiments.value = (await api("/embodiments")).items || [];
  robotwinModels.value = (await api("/dataset-sources/robotwin")).models || [];
  sources.value = (await api("/weights/sources")).items || [];
}

async function resolveRevision() {
  try { revision.value = (await api(`/weights/hub-revision?repo_id=${encodeURIComponent(repoId.value)}`)).revision; }
  catch (e: any) { err.value = e.message; }
}
async function download() {
  acquiring.value = true; err.value = "";
  try {
    await api("/weights/download", { method: "POST", body: JSON.stringify({
      name: selectedSource()?.label, repo_id: repoId.value, revision: revision.value,
      embodiment_id: selectedSource()?.embodiment_id,
    }) });
    await load();
  } catch (e: any) { err.value = e.message; }
  finally { acquiring.value = false; }
}
async function retryDownload(item: any) {
  acquiring.value = true; err.value = "";
  try { await api(`/weights/${item.id}/retry-download`, { method: "POST", body: "{}" }); await load(); }
  catch (e: any) { err.value = e.message; }
  finally { acquiring.value = false; }
}

function gb(n: number) {
  if (!n) return "—";
  return n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB";
}

async function register() {
  err.value = "";
  try {
    await api("/weights", {
      method: "POST",
      body: JSON.stringify({ name: name.value, path: path.value, embodiment_id: embodimentId.value || null }),
    });
    show.value = false;
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">模型</p>
        <h2>模型与血缘</h2>
      </div>
      <button class="primary" type="button" @click="show = !show">＋ 登记基座</button>
    </div>
    <p class="error" role="alert">{{ err }}</p>
    <div class="panel">
      <h3>RoboTwin 模型能力</h3>
      <table>
        <thead><tr><th>模型</th><th>跨本体</th><th>当前直接本体</th><th>RLinf 配方</th></tr></thead>
        <tbody><tr v-for="m in robotwinModels" :key="m.id">
          <td>{{ m.label }}</td>
          <td>{{ m.cross_embodiment === true ? '可适配，需本体映射' : m.cross_embodiment === 'limited' ? '有限' : '否' }}</td>
          <td>{{ m.direct_embodiments?.join(', ') || '无已验证路径' }}</td>
          <td><span class="st-badge" :class="m.ready ? 'st-ok' : 'st-err'">{{ m.ready ? '可配置' : '阻断' }}</span><br><span class="muted xs">{{ m.blocked_reason || m.template_id }}</span></td>
        </tr></tbody>
      </table>
    </div>
    <div class="panel">
      <h3>下载兼容 RLinf 的 π0.5 模型</h3>
      <p>选择与训练本体一致的固定模型源。RoboTwin ALOHA 不能使用 LIBERO/Franka 的归一化资产。</p>
      <label for="model-source">模型源</label>
      <select id="model-source" v-model="repoId" @change="revision = ''"><option v-for="s in sources" :key="s.repo_id" :value="s.repo_id">{{ s.label }}</option></select>
      <label for="model-revision">固定版本（40 位 SHA）</label>
      <input id="model-revision" v-model="revision" placeholder="先读取公开版本" />
      <button type="button" @click="resolveRevision">读取版本</button>
      <button type="button" :disabled="acquiring || !/^[0-9a-f]{40}$/.test(revision)" @click="download">提交下载任务</button>
      <button type="button" @click="load">刷新状态</button>
    </div>
    <div class="panel" v-if="show">
      <h3>登记基座权重</h3>
      <div class="form-grid">
        <span>名称</span><input v-model="name" />
        <span>路径</span><input v-model="path" placeholder="/data/kk/… 或 /var/lib/eb/…" />
        <span>本体</span>
        <select v-model="embodimentId">
          <option value="">（可选）</option>
          <option v-for="e in embodiments" :key="e.id" :value="e.id">{{ e.display_name }}</option>
        </select>
      </div>
      <p class="error">{{ err }}</p>
      <button class="primary" style="margin-top:.6rem" @click="register">保存</button>
    </div>
    <ListControls v-if="items.length" v-model:query="list.query.value" v-model:sort-key="list.sortKey.value" v-model:sort-direction="list.sortDirection.value" v-model:page="list.page.value" v-model:page-size="list.pageSize.value" v-model:text-filters="list.textFilters.value" v-model:multi-filters="list.multiFilters.value" :total="list.total.value" :page-count="list.pageCount.value" :sort-options="sortOptions" :filters="filters" :filter-options="list.filterOptions.value" :active-filter-count="list.activeFilterCount.value" @clear-filters="list.clearFilters" />
    <table v-if="list.total.value">
      <thead>
        <tr><th>名称</th><th>类型</th><th>路径</th><th>体积</th><th>训练 / step</th><th>操作</th></tr>
      </thead>
      <tbody>
        <tr v-for="w in list.visibleItems.value" :key="w.id">
          <td>
            <router-link :to="`/models/${w.id}`">{{ w.name }}</router-link>
            <br><span class="muted xs mono">{{ w.id }}</span>
          </td>
          <td><span class="st-badge" :class="w.kind === 'base' ? 'st-ok' : 'st-run'">{{ w.kind }}</span></td>
          <td class="mono xs">{{ w.path || w.metrics_at_checkpoint?.download?.status }}<p class="error">{{ w.metrics_at_checkpoint?.download?.error }}</p><button v-if="w.kind === 'download_failed'" type="button" :disabled="acquiring" @click="retryDownload(w)">续传</button></td>
          <td class="xs">{{ gb(w.size_bytes) }}</td>
          <td class="xs"><router-link v-if="w.train_job_name" :to="`/train/${w.train_page_id || w.train_job_id}`">{{ w.train_job_name }}</router-link><span v-else>{{ w.train_job_id || "—" }}</span><br v-if="w.train_job_id"><span class="mono muted xs">{{ w.train_job_id }}</span><br v-if="w.step != null">{{ w.step == null ? "" : `step ${w.step}` }}</td>
          <td><router-link v-if="w.card?.schema_version === 'eb/rpo-wbc-model/v1'" :to="{ path: '/eval/new', query: { model_id: w.id } }">创建评估</router-link>
            <DeleteButton :path="`/models/${w.id}`" :name="w.name" :disabled="w.kind === 'downloading'" @deleted="items = items.filter(item => item.id !== w.id)" /></td>
        </tr>
      </tbody>
    </table>
    <p v-else-if="items.length" class="empty-box">没有符合过滤条件的模型。</p>
    <p v-else class="muted">还没有权重。登记基座，或从训练详情勾选 checkpoint。</p>
  </div>
</template>
