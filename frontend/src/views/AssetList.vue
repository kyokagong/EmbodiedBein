<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/assets")).items || []; });
</script>
<template>
  <div>
    <h2>机器人资产</h2>
    <EbTable :columns="[{key:'contract_id',label:'contract'},{key:'mjcf',label:'mjcf'},{key:'usd',label:'usd'},{key:'isaac_task',label:'task'}]" :rows="rows">
      <template #contract_id="{ row }"><router-link :to="`/assets/${row.contract_id}/viewer`">{{ row.contract_id }}</router-link></template>
    </EbTable>
  </div>
</template>
