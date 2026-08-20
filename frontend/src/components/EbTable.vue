<script setup lang="ts">
defineProps<{ columns: { key: string; label: string }[]; rows: Record<string, any>[] }>();
</script>
<template>
  <table v-if="rows?.length">
    <thead>
      <tr><th v-for="c in columns" :key="c.key">{{ c.label }}</th></tr>
    </thead>
    <tbody>
      <tr v-for="(row, i) in rows" :key="row.id || i">
        <td v-for="c in columns" :key="c.key"><slot :name="c.key" :row="row">{{ row[c.key] }}</slot></td>
      </tr>
    </tbody>
  </table>
  <p v-else class="muted">暂无数据</p>
</template>
