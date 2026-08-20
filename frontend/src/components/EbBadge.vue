<script setup lang="ts">
defineProps<{ text: string; tone?: "ok" | "warn" | "bad" }>();
</script>
<template>
  <span class="badge" :class="tone">{{ text }}</span>
</template>
<style scoped>
.ok { color: var(--green); }
.warn { color: var(--amber); }
.bad { color: var(--red); }
</style>
