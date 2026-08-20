<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { VueFlow } from "@vue-flow/core";
import "@vue-flow/core/dist/style.css";
import "@vue-flow/core/dist/theme-default.css";
import { api } from "../api/client";
import EmptyState from "../components/EmptyState.vue";

const route = useRoute();
const edges = ref<any[]>([]);
onMounted(async () => {
  edges.value = (await api(`/models/${route.params.id}/lineage`)).edges || [];
});

const nodes = computed(() => {
  const seen = new Map<string, { id: string; label: string }>();
  for (const e of edges.value) {
    const s = `${e.src_type}:${e.src_id}`;
    const d = `${e.dst_type}:${e.dst_id}`;
    seen.set(s, { id: s, label: s });
    seen.set(d, { id: d, label: d });
  }
  return [...seen.values()].map((n, i) => ({
    id: n.id,
    position: { x: 40 + (i % 3) * 220, y: 40 + Math.floor(i / 3) * 90 },
    data: { label: n.label },
  }));
});
const flowEdges = computed(() =>
  edges.value.map((e, i) => ({
    id: `e${i}`,
    source: `${e.src_type}:${e.src_id}`,
    target: `${e.dst_type}:${e.dst_id}`,
    label: e.relation,
  })),
);
</script>
<template>
  <div>
    <h2>血缘</h2>
    <EmptyState v-if="!edges.length" title="暂无血缘边">训练认领产物后会出现 run→model。</EmptyState>
    <div v-else style="height: 360px; border: 1px solid var(--border, #233); border-radius: 8px">
      <VueFlow :nodes="nodes" :edges="flowEdges" fit-view-on-init />
    </div>
  </div>
</template>
