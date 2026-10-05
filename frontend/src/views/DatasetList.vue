<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
import { useListView } from "../composables/useListView";

const rows = ref<any[]>([]);
const sortOptions = [
  { key: "name", label: "名称", value: (r: any) => r.name },
  { key: "status", label: "状态", value: (r: any) => r.status || "ready" },
  { key: "size", label: "规模", value: (r: any) => Number(r.total_bytes || r.bytes || 0) },
  { key: "source", label: "来源", value: (r: any) => r.source_kind || "path" },
];
const filters = [
  { key: "name", label: "名称 / ID", type: "text" as const, value: (r: any) => `${r.name || ""} ${r.id || ""}` },
  { key: "usage", label: "用途", type: "text" as const, value: (r: any) => r.usage === "eval" ? "评估集 eval" : "训练集 train" },
  { key: "embodiment", label: "本体", type: "text" as const, value: (r: any) => r.embodiment_name },
  { key: "environment", label: "本体环境", type: "text" as const, value: (r: any) => r.embodiment_env_id },
  { key: "source", label: "来源", type: "text" as const, value: (r: any) => `${r.source_kind || "path"} ${r.meta?.native_embodiment || ""}` },
  { key: "path", label: "路径", type: "text" as const, value: (r: any) => r.dest_path || r.source_uri || r.meta?.root },
  { key: "status", label: "状态", type: "multi" as const, value: (r: any) => r.status || "ready" },
];
const list = useListView(rows, sortOptions, (r) => [r.name, r.id, r.usage, r.embodiment_name, r.embodiment_env_id, r.source_kind, r.status, r.dest_path, r.source_uri].join(" "), filters);
onMounted(async () => {
  rows.value = (await api("/datasets")).items || [];
});

function gb(bytes: number | null) {
  if (bytes == null || bytes === 0) return "—";
  return (Number(bytes) / 1e9).toFixed(1) + " GB";
}

function st(s: string) {
  if (s === "ready") return "st-badge st-ok";
  if (s === "downloading" || s === "pending" || s === "extracting") return "st-badge st-run";
  if (s === "failed") return "st-badge st-err";
  return "st-badge";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">数据</p>
        <h2>数据集</h2>
      </div>
      <router-link class="badge" to="/data/new">＋ 接入数据集</router-link>
    </div>
    <ListControls v-if="rows.length" v-model:query="list.query.value" v-model:sort-key="list.sortKey.value" v-model:sort-direction="list.sortDirection.value" v-model:page="list.page.value" v-model:page-size="list.pageSize.value" v-model:text-filters="list.textFilters.value" v-model:multi-filters="list.multiFilters.value" :total="list.total.value" :page-count="list.pageCount.value" :sort-options="sortOptions" :filters="filters" :filter-options="list.filterOptions.value" :active-filter-count="list.activeFilterCount.value" @clear-filters="list.clearFilters" />
    <table v-if="list.total.value">
      <thead>
        <tr>
          <th>名称</th><th>用途</th><th>本体 / 本体环境</th><th>来源</th><th>路径</th><th>规模</th><th>状态</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in list.visibleItems.value" :key="r.id">
          <td>
            <router-link :to="`/data/${r.id}`">{{ r.name }}</router-link><br>
            <span class="muted xs">{{ r.id }}</span>
          </td>
          <td><span class="badge">{{ r.usage === 'eval' ? '评估集' : '训练集' }}</span></td>
          <td>{{ r.embodiment_name || "未关联" }}<br><span class="mono xs muted">{{ r.embodiment_env_id || "—" }}</span></td>
          <td><span class="badge">{{ r.source_kind || "path" }}</span><br><span v-if="r.meta?.native_embodiment" class="muted xs">{{ r.meta.native_embodiment }} · {{ r.meta.compatibility }}</span></td>
          <td class="mono xs">{{ r.dest_path || r.source_uri || r.meta?.root || "—" }}</td>
          <td>{{ gb(r.total_bytes || r.bytes) }}</td>
          <td><span :class="st(r.status || 'ready')">{{ r.status || "ready" }}</span></td>
        </tr>
      </tbody>
    </table>
    <p v-else-if="rows.length" class="empty-box">没有符合过滤条件的数据集。</p>
    <p v-else class="muted">无数据集。接入本地路径、Hugging Face 或 LIBERO 测试场景。</p>
  </div>
</template>
