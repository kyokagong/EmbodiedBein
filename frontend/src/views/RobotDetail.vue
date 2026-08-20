<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
const route = useRoute();
const robot = ref<any>(null);
const err = ref("");
const series = ref<{step:number;value:number}[]>([]);
onMounted(async () => {
  const robots = (await api("/robots")).items || [];
  robot.value = robots.find((r: any) => r.id === route.params.id) || robots[0];
});
async function promote() {
  try {
    await api(`/releases/${route.params.id}/promote`, { method: "POST", headers: { "X-Confirm": String(route.params.id) } });
  } catch (e: any) { err.value = e.message; }
}
</script>
<template>
  <div>
    <h2>机器人 {{ robot?.name || route.params.id }}</h2>
    <p class="muted">放行必须人按物理确认。平台 API 即使带 X-Confirm 仍 403。</p>
    <button @click="promote">请求放行（会 403）</button>
    <p class="error">{{ err }}</p>
    <Chart title="robot.temp_c" :points="series" />
    <p class="muted" v-if="!series.length">无遥测点 — 不会用 Math.random() 造温度</p>
  </div>
</template>
