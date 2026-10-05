<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const cases = ref<any[]>([]); const error = ref("");
onMounted(async () => {
  try {
    const reports = (await api("/evals/reports")).items || [];
    const results = await Promise.all(reports.map(async (report: any) => {
      try {
        const data = await api(`/evals/reports/${report.id}/episodes?failures_only=true`);
        return (data.items || []).map((ep: any) => ({ ...ep, report_id: report.id }));
      } catch { return []; }
    }));
    cases.value = results.flat();
  } catch (e: any) { error.value = e.message; }
});
</script>
<template>
  <div><h2>失败案例</h2><p class="muted">按实际 episode 索引；总体高分报告中的失败也会列出。</p>
    <p v-if="error" class="error">{{ error }}</p>
    <table v-if="cases.length"><thead><tr><th>案例</th><th>任务</th><th>初始状态</th><th>状态</th><th>报告</th></tr></thead>
      <tbody><tr v-for="c in cases" :key="c.report_id + ':' + c.case_id"><td class="mono xs">{{ c.case_id }}</td><td>{{ c.task_id }}</td><td>{{ c.init_state_id }}</td><td>{{ c.status }}</td><td><router-link :to="`/eval/reports/${c.report_id}`">查看报告</router-link></td></tr></tbody></table>
    <p v-else-if="!error" class="muted">没有可索引的失败案例。历史汇总报告若无逐案例产物，不会被推断成零失败。</p>
  </div>
</template>
