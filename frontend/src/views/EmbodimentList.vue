<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import ListControls from "../components/ListControls.vue";
import { useListView } from "../composables/useListView";

const items = ref<any[]>([]);
const sortOptions = [
  { key: "name", label: "名称", value: (e: any) => e.display_name },
  { key: "variants", label: "URDF 数", value: (e: any) => Number(e.variant_count || e.urdf_variants?.length || 0) },
  { key: "tasks", label: "任务数", value: (e: any) => Number(e.task_count || e.gym_ids?.length || 0) },
];
const filters = [
  { key: "name", label: "名称 / ID", type: "text" as const, value: (e: any) => `${e.display_name || ""} ${e.id || ""}` },
  { key: "path", label: "路径", type: "text" as const, value: (e: any) => e.root_path || e.source?.path },
  { key: "gym", label: "Gym 任务", type: "text" as const, value: (e: any) => (e.gym_ids || []).join(" ") },
];
const list = useListView(items, sortOptions, (e) => [e.display_name, e.id, e.root_path, e.source?.path, ...(e.gym_ids || [])].join(" "), filters);
onMounted(async () => {
  items.value = (await api("/embodiments")).items || [];
});
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">本体</p>
        <h2>本体 <span class="badge">{{ items.length }}</span></h2>
      </div>
      <router-link class="badge" to="/embodiments/new">登记路径</router-link>
    </div>
    <ListControls v-if="items.length" v-model:query="list.query.value" v-model:sort-key="list.sortKey.value" v-model:sort-direction="list.sortDirection.value" v-model:page="list.page.value" v-model:page-size="list.pageSize.value" v-model:text-filters="list.textFilters.value" v-model:multi-filters="list.multiFilters.value" :total="list.total.value" :page-count="list.pageCount.value" :sort-options="sortOptions" :filters="filters" :filter-options="list.filterOptions.value" :active-filter-count="list.activeFilterCount.value" @clear-filters="list.clearFilters" />
    <div v-if="list.total.value" class="grid2">
      <router-link class="card card-link" v-for="e in list.visibleItems.value" :key="e.id" :to="`/embodiments/${e.id}`">
        <div class="l mono">{{ e.id }}</div>
        <div class="v">{{ e.display_name }}</div>
        <p class="muted xs">{{ e.root_path || e.source?.path || "—" }}</p>
        <p class="xs" style="margin-top:.4rem">
          <span class="badge">{{ e.variant_count || (e.urdf_variants || []).length }} URDF</span>
          <span class="badge">{{ e.task_count || (e.gym_ids || []).length }} gym</span>
        </p>
      </router-link>
    </div>
    <p v-else-if="items.length" class="empty-box">没有符合过滤条件的本体。</p>
    <p v-if="!items.length" class="muted">还没有本体。登记一份本地代码路径即可扫描 URDF。</p>
  </div>
</template>
