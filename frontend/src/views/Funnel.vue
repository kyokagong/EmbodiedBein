<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const funnel = ref<any>(null);
onMounted(async () => {
  const ds = await api(`/datasets/${route.params.id}`);
  const vid = ds.versions?.[0]?.id;
  if (vid) funnel.value = await api(`/datasets/${route.params.id}/versions/${vid}/funnel`);
});
</script>
<template>
  <div>
    <h2>数据漏斗</h2>
    <div class="card" v-for="s in funnel?.stages || []" :key="s.name">
      <div class="l">{{ s.name }}</div>
      <div class="v">{{ s.n }}</div>
    </div>
  </div>
</template>
