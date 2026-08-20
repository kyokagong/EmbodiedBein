<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
import EbBadge from "../components/EbBadge.vue";

const q = ref<any>({ items: [] });
const gpus = ref<any>(null);
let ws: WebSocket | null = null;
onMounted(async () => {
  q.value = await api("/resources/queue");
  gpus.value = await api("/resources/gpus");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/queue`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "queue") q.value = msg.payload;
    } catch {}
  };
});
onUnmounted(() => ws?.close());
</script>
<template>
  <div>
    <h2>队列与 GPU</h2>
    <div class="kpis">
      <div class="card" v-for="g in gpus?.gpus || []" :key="g.index">
        <div class="l">GPU {{ g.index }} {{ g.name }}</div>
        <div class="v">{{ g.run_id ? "占用" : "空闲" }}</div>
        <div class="muted">{{ g.memory_free_mb }} / {{ g.memory_total_mb }} MB · {{ g.run_id || "idle" }}</div>
      </div>
    </div>
    <p class="muted" v-if="!(gpus?.gpus || []).length">ebd doctor 未看到 GPU</p>
    <EbTable
      :columns="[{key:'id',label:'job'},{key:'kind',label:'kind'},{key:'status',label:'status'},{key:'priority',label:'priority'}]"
      :rows="q?.items || []"
    >
      <template #status="{ row }"><EbBadge :text="row.status" /></template>
    </EbTable>
  </div>
</template>
