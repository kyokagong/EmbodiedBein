<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
import { useListView } from "../composables/useListView";

const items = ref<any[]>([]);
const error = ref("");
const sortOptions = [
  { key: "created", label: "创建时间", value: (j: any) => j.created_at },
  { key: "name", label: "任务名称", value: (j: any) => j.name },
  { key: "status", label: "执行状态", value: (j: any) => j.status },
  { key: "trials", label: "有效样本", value: (j: any) => Number(j.metrics_summary?.n_trials || 0) },
];
const filters = [
  { key: "name", label: "任务 / ID", type: "text" as const, value: (j: any) => `${j.name || ""} ${j.id || ""}` },
  { key: "checkpoint", label: "Checkpoint", type: "text" as const, value: (j: any) => `${j.checkpoint?.run_id || "历史模型"} ${j.checkpoint?.step ?? ""}` },
  { key: "suite", label: "套件", type: "text" as const, value: (j: any) => `${j.suite_id || "历史套件"} ${j.plan?.profile || ""}` },
  { key: "status", label: "执行状态", type: "multi" as const, value: (j: any) => j.status },
  { key: "qualification", label: "资格", type: "multi" as const, value: (j: any) => j.metrics_summary?.qualification || "待报告" },
  { key: "created", label: "创建时间", type: "text" as const, value: (j: any) => j.created_at },
];
const list = useListView(items, sortOptions, (j) => [j.name, j.id, j.status, j.suite_id, j.checkpoint?.run_id, j.metrics_summary?.qualification].join(" "), filters, "created", "desc");
onMounted(async () => {
  try { items.value = (await api("/evals/jobs")).items || []; }
  catch (e: any) { error.value = e.message; }
});
</script>
<template>
  <div>
    <div class="page-head"><div><p class="crumb">评估</p><h2>评估任务</h2></div><router-link class="badge" to="/eval/new">＋ 新建评估</router-link></div>
    <p v-if="error" class="error">{{ error }}</p>
    <ListControls v-if="items.length" v-model:query="list.query.value" v-model:sort-key="list.sortKey.value" v-model:sort-direction="list.sortDirection.value" v-model:page="list.page.value" v-model:page-size="list.pageSize.value" v-model:text-filters="list.textFilters.value" v-model:multi-filters="list.multiFilters.value" :total="list.total.value" :page-count="list.pageCount.value" :sort-options="sortOptions" :filters="filters" :filter-options="list.filterOptions.value" :active-filter-count="list.activeFilterCount.value" @clear-filters="list.clearFilters" />
    <table v-if="list.total.value"><thead><tr><th>任务</th><th>checkpoint</th><th>套件</th><th>执行状态</th><th>有效样本</th><th>资格</th><th>创建</th></tr></thead>
      <tbody><tr v-for="j in list.visibleItems.value" :key="j.id">
        <td><router-link :to="`/eval/jobs/${j.id}`">{{ j.name }}</router-link><br><span class="muted xs mono">{{ j.id }}</span></td>
        <td class="mono xs">{{ j.model_version_id || j.checkpoint?.run_id || '历史模型' }}<br>step {{ j.plan?.checkpoint_step ?? j.checkpoint?.step ?? '—' }}</td>
        <td>{{ j.suite_id || '历史套件' }}<br><span v-if="j.plan" class="muted xs">{{ j.plan.profile }} · 计划 {{ j.plan.planned }} 条</span></td>
        <td>{{ j.status }}</td><td>{{ j.metrics_summary?.n_trials ?? '—' }} / {{ j.plan?.planned ?? '—' }}</td>
        <td>{{ j.metrics_summary?.qualification || '待报告' }}</td><td class="muted xs">{{ (j.created_at || '').replace('T',' ').slice(0,16) }}</td>
      </tr></tbody></table>
    <p v-else-if="items.length" class="empty-box">没有符合过滤条件的评估任务。</p>
    <p v-else-if="!error" class="empty-box">暂无评估任务。可从训练 checkpoint 进入，也可新建评估。</p>
  </div>
</template>
