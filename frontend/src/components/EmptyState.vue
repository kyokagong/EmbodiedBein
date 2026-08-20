<script setup lang="ts">
defineProps<{ title?: string }>();
</script>
<template>
  <div class="card">
    <div class="l">{{ title || "空" }}</div>
    <p class="muted"><slot>暂无数据</slot></p>
  </div>
</template>
