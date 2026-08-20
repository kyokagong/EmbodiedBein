<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const sid = ref("");
const text = ref("帮我看看队列和显存");
const log = ref<{ role: string; text: string }[]>([]);
onMounted(async () => {
  const s = await api("/agent/sessions", { method: "POST", body: "{}" });
  sid.value = s.id;
});
async function send() {
  const r = await api(`/agent/sessions/${sid.value}/messages`, {
    method: "POST",
    body: JSON.stringify({ text: text.value }),
  });
  log.value.push({ role: "user", text: text.value }, { role: "agent", text: r.reply });
}
</script>
<template>
  <div>
    <h2>Agent 控制台</h2>
    <p class="muted">规则优先。永久禁止：真机放行 / 改契约 / 门禁豁免。</p>
    <div class="card" v-for="(m, i) in log" :key="i">
      <div class="l">{{ m.role }}</div>
      <div>{{ m.text }}</div>
    </div>
    <div class="row">
      <input v-model="text" style="flex:1" />
      <button @click="send">发送</button>
    </div>
  </div>
</template>
