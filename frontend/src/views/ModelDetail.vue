<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { VueFlow } from "@vue-flow/core";
import "@vue-flow/core/dist/style.css";
import "@vue-flow/core/dist/theme-default.css";
import { api } from "../api/client";

const route = useRoute();
const data = ref<any>(null);
const edges = ref<any[]>([]);

onMounted(async () => {
  data.value = await api(`/models/${route.params.id}`);
  edges.value = data.value.edges || (await api(`/models/${route.params.id}/lineage`)).edges || [];
});

const nodes = computed(() => {
  const seen = new Map<string, { id: string; label: string }>();
  seen.set(`model:${route.params.id}`, { id: `model:${route.params.id}`, label: data.value?.name || String(route.params.id) });
  for (const e of edges.value) {
    const s = `${e.src_type}:${e.src_id}`;
    const d = `${e.dst_type}:${e.dst_id}`;
    seen.set(s, { id: s, label: s });
    seen.set(d, { id: d, label: d });
  }
  return [...seen.values()].map((n, i) => ({
    id: n.id,
    position: { x: 40 + (i % 3) * 240, y: 40 + Math.floor(i / 3) * 100 },
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
    <div class="page-head">
      <div>
        <p class="crumb">模型 / {{ route.params.id }}</p>
        <h2>{{ data?.name || route.params.id }}</h2>
      </div>
      <router-link class="badge" to="/models">← 列表</router-link>
    </div>
    <p class="muted" v-if="data">
      {{ data.kind || data.provenance || "model" }} · {{ data.path || data.card?.source_path || "—" }}
    </p>
    <div class="kpis" v-if="data">
      <div class="card"><div class="l">类型</div><div class="v">{{ data.kind || "version" }}</div></div>
      <div class="card"><div class="l">本体</div><div class="v">{{ data.embodiment_id || data.embodiment_env_id || "—" }}</div></div>
      <div class="card"><div class="l">训练</div><div class="v">{{ data.train_job_id || "—" }}</div></div>
      <div class="card"><div class="l">父模型</div><div class="v">{{ data.parent_model_id || "—" }}</div></div>
    </div>
    <div class="panel">
      <h3>DAG 血缘</h3>
      <p class="muted xs">本体 + 数据 + 基座 + 训练 → 产物</p>
      <div style="height: 360px; border: 1px solid var(--line); border-radius: 8px">
        <VueFlow :nodes="nodes" :edges="flowEdges" fit-view-on-init />
      </div>
    </div>
  </div>
</template>
