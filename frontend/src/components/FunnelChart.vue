<script setup lang="ts">
defineProps<{ stages: { name: string; n: number }[] }>();
function maxN(stages: { n: number }[]) {
  return Math.max(1, ...stages.map((s) => s.n || 0));
}
</script>
<template>
  <div>
    <div class="funnel-row" v-for="s in stages" :key="s.name">
      <span>{{ s.name }}</span>
      <div class="funnel-track">
        <div
          class="funnel-fill"
          :class="{ zero: !s.n }"
          :style="{ width: ((s.n || 0) / maxN(stages)) * 100 + '%' }"
        />
      </div>
      <span class="mono">{{ s.n }}</span>
    </div>
  </div>
</template>
