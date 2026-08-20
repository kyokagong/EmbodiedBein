<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => {
  const all = (await api("/evals/reports")).items || [];
  rows.value = all.filter((r: any) => (r.metrics?.success_rate ?? 1) < 0.5);
});
</script>
<template>
  <div>
    <h2>失败库</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'suite_id',label:'suite'}]" :rows="rows" />
    <p class="muted" v-if="!rows.length">没有失败报告</p>
  </div>
</template>
