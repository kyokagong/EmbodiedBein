<script setup lang="ts">
import { ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import EmptyState from "../components/EmptyState.vue";
import LineageGraph from "../components/LineageGraph.vue";

const route = useRoute();
const edges = ref<any[]>([]);
const nodes = ref<any[]>([]);
watch(() => route.params.id, async id => {
  edges.value = [];
  nodes.value = [];
  const lineage = await api(`/models/${id}/lineage`);
  if (route.params.id !== id) return;
  edges.value = lineage.edges || [];
  nodes.value = lineage.nodes || [];
}, { immediate: true });
</script>
<template>
  <div>
    <h2>血缘</h2>
    <EmptyState v-if="!edges.length" title="暂无血缘边">训练认领产物后会出现 run→model。</EmptyState>
    <LineageGraph v-else :nodes="nodes" :edges="edges" />
  </div>
</template>
