<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
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

onMounted(async () => {
  try {
    items.value = (await api("/train")).items || [];
  } catch (e: any) {
    err.value = e.message;
  }
});

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
          <th>名称</th><th>产出模型</th><th>状态</th><th>模板</th><th>本体</th><th>GPU</th><th>Ray</th><th>创建</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="j in list.visibleItems.value" :key="j.id">
          <td>
            <router-link :to="`/train/${j.id}`">{{ j.name }}</router-link>
            <br><span class="muted xs mono">{{ j.id }}</span>
          </td>
          <td>{{ j.model_name || '—' }}</td>
          <td><span :class="st(j.status)">{{ j.status }}</span></td>
          <td class="mono xs">{{ j.template_id }}</td>
          <td class="mono xs">{{ j.embodiment_id }}</td>
          <td class="xs">{{ (j.gpus_allocated || []).join(", ") || "—" }}</td>
          <td class="mono xs">{{ j.ray_port || "—" }}</td>
          <td class="xs muted">{{ (j.created_at || "").replace("T", " ").slice(0, 16) }}</td>
        </tr>
      </tbody>
    </table>
    <div v-else-if="items.length" class="empty-box">没有符合过滤条件的训练任务。</div>
    <div v-else class="empty-box">
      <b>还没有训练任务</b>
      本版只提交 RLinf 三模板：WBC PPO / Reach SFT / Collect。
    </div>
  </div>
</template>
