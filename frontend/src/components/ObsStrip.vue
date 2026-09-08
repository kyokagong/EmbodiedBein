<script setup lang="ts">
defineProps<{
  items: { term: string; dim: number; color: string; index_start: number; index_end: number; source?: string }[];
}>();
</script>
<template>
  <div>
    <div class="obs-strip">
      <span
        v-for="t in items"
        :key="t.term"
        :style="{ flex: Math.max(t.dim, 1), background: t.color }"
        :title="`${t.term} · ${t.source || ''} · ${t.index_start}–${t.index_end}`"
      >{{ t.term }} {{ t.dim }}</span>
    </div>
    <div class="obs-legend">
      <span v-for="t in items" :key="t.term + 'l'">
        <i :style="{ background: t.color }"></i>{{ t.index_start }}–{{ t.index_end }} {{ t.term }} · {{ t.source }}
      </span>
    </div>
  </div>
</template>
