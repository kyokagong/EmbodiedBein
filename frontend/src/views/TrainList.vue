<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
import DeleteButton from "../components/DeleteButton.vue";
import { useListView } from "../composables/useListView";

const items = ref<any[]>([]);
const err = ref("");
const sortOptions = [
  { key: "created", label: "创建时间", value: (j: any) => j.created_at },
  { key: "name", label: "名称", value: (j: any) => j.name },
  { key: "model_name", label: "产出模型", value: (j: any) => j.model_name },
  { key: "status", label: "状态", value: (j: any) => j.status },
  { key: "template", label: "模板", value: (j: any) => j.template_id },
];
const filters = [
  { key: "name", label: "名称 / ID", type: "text" as const, value: (j: any) => `${j.name || ""} ${j.id || ""}` },
  { key: "model_name", label: "产出模型", type: "text" as const, value: (j: any) => j.model_name },
  { key: "status", label: "状态", type: "multi" as const, value: (j: any) => j.status },
  { key: "template", label: "模板", type: "text" as const, value: (j: any) => j.template_id },
  { key: "embodiment", label: "本体", type: "text" as const, value: (j: any) => j.embodiment_id },
  { key: "gpu", label: "GPU", type: "text" as const, value: (j: any) => (j.gpus_allocated || []).join(", ") },
  { key: "ray", label: "Ray", type: "text" as const, value: (j: any) => String(j.ray_port || "") },
  { key: "created", label: "创建时间", type: "text" as const, value: (j: any) => j.created_at },
];
const list = useListView(items, sortOptions, (j) => [j.name, j.model_name, j.id, j.status, j.template_id, j.embodiment_id, ...(j.gpus_allocated || []), j.ray_port].join(" "), filters, "created", "desc");

let timer: number | undefined;
let controller: AbortController | undefined;
let disposed = false;
let generation = 0;
onUnmounted(() => { disposed = true; generation++; clearTimeout(timer); controller?.abort(); });
async function load() {
  clearTimeout(timer);
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal, current = ++generation;
  try {
    const rows = (await api("/train", { signal })).items || [];
    const results = await Promise.allSettled(rows.filter((row: any) => row.kind === "native_rl" &&
      row.run_id && ["running", "preparing"].includes(row.status)).map(async (row: any) => {
      const data = await api(`/runs/${row.run_id}/metrics?tag=distributed%2Foptimizer_steps`, { signal });
      const points = data.points || data.series?.["distributed/optimizer_steps"] || [];
      if (points.length) row.live_updates = points[points.length - 1].value;
    }));
    if (disposed || current !== generation) return;
    items.value = rows;
    err.value = results.some(result => result.status === "rejected") ? "部分实时指标暂不可用，稍后重试。" : "";
  } catch (e: any) {
    if (!disposed && current === generation && e.name !== "AbortError") err.value = e.message;
  } finally {
    if (!disposed && current === generation) timer = window.setTimeout(load, 8000);
  }
}
onMounted(load);

function st(s: string) {
  if (s === "running" || s === "preparing") return "st-badge st-run";
  if (s === "succeeded" || s === "ready") return "st-badge st-ok";
  if (s === "queued") return "st-badge st-q";
  if (s === "failed" || s === "aborted") return "st-badge st-err";
  return "st-badge";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">训练</p>
        <h2>训练任务</h2>
      </div>
      <router-link class="badge" to="/train/new">＋ 新建 RLinf 训练</router-link>
    </div>
    <p class="error">{{ err }}</p>
    <ListControls v-if="items.length" v-model:query="list.query.value" v-model:sort-key="list.sortKey.value" v-model:sort-direction="list.sortDirection.value" v-model:page="list.page.value" v-model:page-size="list.pageSize.value" v-model:text-filters="list.textFilters.value" v-model:multi-filters="list.multiFilters.value" :total="list.total.value" :page-count="list.pageCount.value" :sort-options="sortOptions" :filters="filters" :filter-options="list.filterOptions.value" :active-filter-count="list.activeFilterCount.value" @clear-filters="list.clearFilters" />
    <table v-if="list.total.value">
      <thead>
        <tr>
          <th>名称</th><th>产出模型</th><th>状态 / 实际更新</th><th>模板</th><th>本体</th><th>GPU</th><th>Ray</th><th>创建</th><th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="j in list.visibleItems.value" :key="j.id">
          <td>
            <router-link :to="`/train/${j.id}`">{{ j.name }}</router-link>
            <br><span class="muted xs mono">{{ j.id }}</span>
          </td>
          <td><router-link v-if="j.metrics_summary?.model_version_id" :to="`/models/${j.metrics_summary.model_version_id}`">{{ j.model_name || j.metrics_summary.model_version_id }}</router-link><span v-else>{{ j.model_name || '—' }}</span></td>
          <td><span :class="st(j.status)">{{ j.status }}</span><p v-if="j.kind === 'native_rl'" class="xs">{{ j.metrics_summary?.optimizer_steps ?? j.live_updates ?? '等待实测' }} / {{ j.params?.native_train_plan?.target_updates }} 更新</p></td>
          <td class="mono xs">{{ j.template_id }}</td>
          <td class="mono xs">{{ j.embodiment_id }}</td>
          <td class="xs">{{ (j.gpus_allocated || []).join(", ") || "—" }}</td>
          <td class="mono xs">{{ j.ray_port || "—" }}</td>
          <td class="xs muted">{{ (j.created_at || "").replace("T", " ").slice(0, 16) }}</td>
          <td><span v-if="j.record_kind === 'pbfm_validation'" class="muted xs">{{ j.training_mode === 'training_long_recovery' ? '100000故障恢复' : j.training_mode === 'training_long_exploration' ? '单次100000探索' : j.training_mode === 'training_exploration' ? '单次10000探索' : '四卡分段验证' }}</span><DeleteButton v-else :path="`/train/${j.id}`" :name="j.name" :disabled="!['succeeded','failed','timeout','cancelled','infra_failed'].includes(j.status)" @deleted="items = items.filter(item => item.id !== j.id)" /></td>
        </tr>
      </tbody>
    </table>
    <div v-else-if="items.length" class="empty-box">没有符合过滤条件的训练任务。</div>
    <div v-else class="empty-box">
      <b>还没有训练任务</b>
      从“新建 RLinf 训练”选择兼容模板、训练环境与数据。
    </div>
  </div>
</template>
