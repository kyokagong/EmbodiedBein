<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const data = ref<any>(null);
onMounted(async () => {
  data.value = await api(`/shares/${route.params.token}`);
});
</script>
<template>
  <div class="main">
    <p class="muted">只读分享 · Showcase · 无需登录</p>
    <h2>{{ data?.title || "Showcase" }}</h2>
    <div class="kpis" v-if="data?.snapshot?.kpis">
      <div class="card" v-for="k in data.snapshot.kpis" :key="k.label">
        <div class="l">{{ k.label }}</div>
        <div class="v">{{ k.value }}</div>
        <div class="muted">{{ k.foot }}</div>
      </div>
    </div>
    <ul>
      <li v-for="m in data?.snapshot?.milestones || []" :key="m.t"><b>{{ m.t }}</b> {{ m.d }}</li>
    </ul>
    <p class="muted">{{ data?.snapshot?.caveat }}</p>
  </div>
</template>
