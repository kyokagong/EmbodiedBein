<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import GatePanel from "../components/GatePanel.vue";
const route = useRoute();
const row = ref<any>(null);
onMounted(async () => { row.value = await api(`/evals/reports/${route.params.id}`); });
</script>
<template>
  <div>
    <h2>评估报告</h2>
    <p>success_rate {{ row?.metrics?.success_rate ?? "—" }}</p>
    <p class="muted">per-seed {{ row?.metrics?.per_seed }}</p>
    <p class="muted">无录像时显示「本 run 未开 camera」，不用假缩略图。</p>
    <GatePanel :result="row?.gate_results" />
  </div>
</template>
