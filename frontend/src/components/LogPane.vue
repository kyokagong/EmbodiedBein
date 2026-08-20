<script setup lang="ts">
defineProps<{ lines: string[] }>();
</script>
<template>
  <pre class="logpane">{{ (lines || []).join("\n") || "暂无日志" }}</pre>
</template>
<style scoped>
.logpane { max-height: 320px; overflow: auto; font-size: .75rem; }
</style>
