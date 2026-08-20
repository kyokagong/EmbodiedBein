<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/evals/reports")).items || []; });
</script>
<template>
  <div>
    <h2>评估对比</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'suite_id',label:'suite'},{key:'model_version_id',label:'model'}]" :rows="rows">
      <template #id="{ row }"><router-link :to="`/eval/reports/${row.id}`">{{ row.id }}</router-link></template>
    </EbTable>
  </div>
</template>
