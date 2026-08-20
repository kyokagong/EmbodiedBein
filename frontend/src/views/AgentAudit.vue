<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/agent/audit")).items || []; });
</script>
<template>
  <div>
    <h2>Agent 审计</h2>
    <EbTable :columns="[{key:'role',label:'role'},{key:'tool',label:'tool'},{key:'guardrail_result',label:'guardrail'}]" :rows="rows" />
  </div>
</template>
