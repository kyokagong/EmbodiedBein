<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const data = ref<any>(null);
const metrics = ref<any>(null);
onMounted(async () => {
  data.value = await api(`/runs/${route.params.id}`);
  metrics.value = await api(`/runs/${route.params.id}/metrics`);
});
function spark(pts: { step: number; value: number }[]) {
  if (!pts?.length) return "";
  const w = 320, h = 64;
  const xs = pts.map((p) => p.step);
  const ys = pts.map((p) => p.value);
  const minX = Math.min(...xs), maxX = Math.max(...xs) || 1;
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  return pts
    .map((p, i) => {
      const x = ((p.step - minX) / spanX) * w;
      const y = h - ((p.value - minY) / spanY) * h;
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}
</script>
<template>
  <div>
    <h2>Run {{ data?.id }}</h2>
    <p class="muted">{{ data?.status }} · {{ data?.abort_reason || "ok" }} · provenance {{ data?.provenance }}</p>
    <div class="card" v-for="(pts, tag) in metrics?.series || {}" :key="tag">
      <div class="l">{{ tag }}</div>
      <svg width="320" height="64" v-if="pts?.length">
        <path :d="spark(pts)" fill="none" stroke="#53d4ff" stroke-width="1.5" />
      </svg>
      <div class="muted" v-else>无序列</div>
    </div>
    <pre>{{ JSON.stringify(data?.metrics_summary && { ...data.metrics_summary, series: undefined }, null, 2) }}</pre>
  </div>
</template>
