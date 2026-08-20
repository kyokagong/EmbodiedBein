<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const me = ref<any>(null);
const contracts = ref<any[]>([]);
const jobs = ref<any[]>([]);
const models = ref<any[]>([]);
const gpus = ref<any>(null);
onMounted(async () => {
  me.value = await api("/auth/me");
  contracts.value = (await api("/contracts")).items || [];
  jobs.value = (await api("/jobs")).items || [];
  models.value = (await api("/models")).versions || [];
  gpus.value = await api("/resources/gpus");
});
</script>
<template>
  <div>
    <h2>总览</h2>
    <div class="kpis">
      <div class="card"><div class="l">身份</div><div class="v">{{ me?.principal || "…" }}</div></div>
      <div class="card"><div class="l">契约</div><div class="v">{{ contracts.length }}</div></div>
      <div class="card"><div class="l">模型</div><div class="v">{{ models.length }}</div></div>
      <div class="card"><div class="l">任务</div><div class="v">{{ jobs.length }}</div></div>
    </div>
    <p class="muted">GPU {{ gpus?.gpus?.length ?? "—" }} · {{ gpus?.hostname || "ebd doctor" }}</p>
    <div class="card">
      <div class="l">最近任务</div>
      <table>
        <thead><tr><th>id</th><th>kind</th><th>status</th><th>contract</th></tr></thead>
        <tbody>
          <tr v-for="j in jobs.slice(0, 8)" :key="j.id">
            <td>{{ j.id }}</td><td>{{ j.kind }}</td><td><span class="badge">{{ j.status }}</span></td><td>{{ j.contract_id }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
