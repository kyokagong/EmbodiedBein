<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbKpi from "../components/EbKpi.vue";
import EbTable from "../components/EbTable.vue";

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
      <EbKpi label="身份" :value="me?.principal || '…'" />
      <EbKpi label="契约" :value="contracts.length" />
      <EbKpi label="模型" :value="models.length" />
      <EbKpi label="队列" :value="jobs.filter(j => j.status==='queued' || j.status==='running').length" />
    </div>
    <p class="muted">GPU {{ gpus?.gpus?.length ?? "—" }} · {{ gpus?.hostname || "ebd doctor" }}</p>
    <EbTable
      :columns="[{key:'id',label:'id'},{key:'kind',label:'kind'},{key:'status',label:'status'},{key:'contract_id',label:'contract'}]"
      :rows="jobs.slice(0, 12)"
    />
  </div>
</template>
