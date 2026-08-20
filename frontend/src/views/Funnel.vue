<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const funnel = ref<any>({});
onMounted(async () => {
  const ds = await api(`/datasets/${route.params.id}`);
  const vid = ds.versions?.[0]?.id;
  if (vid) funnel.value = await api(`/datasets/${route.params.id}/versions/${vid}/funnel`);
});
const layers = ["source", "raw", "valid_episode", "retargeted", "canonical", "slice"];
</script>
<template>
  <div>
    <h2>数据漏斗</h2>
    <p class="muted">拒绝原因来自扫描，不是种子 155000。</p>
    <div class="kpis">
      <div class="card" v-for="name in layers" :key="name">
        <div class="l">{{ name }}</div>
        <div class="v">{{ funnel[name]?.n ?? funnel.stages?.find((s:any)=>s.name===name)?.n ?? "—" }}</div>
        <div class="muted" v-for="r in (funnel[name]?.rejected || [])" :key="r.reason">{{ r.reason }} × {{ r.count }}</div>
      </div>
    </div>
  </div>
</template>
