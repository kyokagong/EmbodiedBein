<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
import LogPane from "../components/LogPane.vue";

const route = useRoute();
const data = ref<any>(null);
const metrics = ref<any>({ series: {} });
const logs = ref<string[]>([]);
let ws: WebSocket | null = null;

onMounted(async () => {
  const id = route.params.id;
  data.value = await api(`/runs/${id}`);
  metrics.value = await api(`/runs/${id}/metrics`);
  logs.value = (await api(`/runs/${id}/logs?tail=200`)).lines || [];
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "metrics" && msg.payload?.series) metrics.value = msg.payload;
    if (msg.type === "log") logs.value = msg.payload.lines || logs.value;
  };
});
onUnmounted(() => ws?.close());
</script>
<template>
  <div>
    <h2>Run {{ data?.id }}</h2>
    <p class="muted">{{ data?.status }} · {{ data?.abort_reason || "ok" }}</p>
    <div v-for="(pts, tag) in metrics?.series || {}" :key="String(tag)">
      <Chart :title="String(tag)" :points="pts" />
    </div>
    <p class="muted" v-if="!Object.keys(metrics?.series || {}).length">尚无 TSDB 点</p>
    <h3>console.log</h3>
    <LogPane :lines="logs" />
  </div>
</template>
