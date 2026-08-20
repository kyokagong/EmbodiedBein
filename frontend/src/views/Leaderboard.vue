<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/evals/leaderboard")).items || []; });
</script>
<template>
  <div>
    <h2>排行榜</h2>
    <p class="muted">来自真实 EvalReport。6 个 checkpoints 策略在 ingest 后出现在模型页，有报告才上榜。</p>
    <EbTable :columns="[{key:'id',label:'id'},{key:'suite_id',label:'suite'},{key:'model_version_id',label:'model'}]" :rows="rows">
      <template #id="{ row }"><router-link :to="`/eval/reports/${row.id}`">{{ row.id }}</router-link></template>
    </EbTable>
  </div>
</template>
