<script setup lang="ts">
defineProps<{ result?: { passed?: boolean; violations?: any[] } }>();
</script>
<template>
  <div class="card">
    <div class="l">门禁</div>
    <div class="v">{{ result?.passed === true ? "通过" : result?.passed === false ? "未通过" : "尚未评估" }}</div>
    <ul v-if="result?.violations?.length">
      <li v-for="(v, i) in result.violations" :key="i">{{ v.path }} 期望 {{ v.expected }} 实际 {{ v.actual }}</li>
    </ul>
  </div>
</template>
