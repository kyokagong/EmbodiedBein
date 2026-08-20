<script setup lang="ts">
import { ref } from "vue";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
const a = ref("");
const b = ref("");
const out = ref<any>(null);
async function go() {
  out.value = await api(`/runs/compare?ids=${a.value},${b.value}`);
}
</script>
<template>
  <div>
    <h2>Run 对比</h2>
    <div class="row">
      <input v-model="a" placeholder="run id A" />
      <input v-model="b" placeholder="run id B" />
      <button @click="go">对比</button>
    </div>
    <p v-if="out" class="muted">{{ out.label }}</p>
    <ul v-if="out?.diff">
      <li v-for="d in out.diff" :key="d.path">{{ d.path }}: {{ d.a }} → {{ d.b }}</li>
    </ul>
    <div v-for="(series, id) in out?.series || {}" :key="String(id)">
      <Chart v-if="series['Train/mean_reward']" :title="String(id)" :points="series['Train/mean_reward']" />
    </div>
  </div>
</template>
