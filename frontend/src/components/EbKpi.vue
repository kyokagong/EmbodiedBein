<script setup lang="ts">
defineProps<{ label: string; value: string | number; foot?: string }>();
</script>
<template>
  <div class="card">
    <div class="l">{{ label }}</div>
    <div class="v">{{ value }}</div>
    <div class="muted" v-if="foot">{{ foot }}</div>
  </div>
</template>
