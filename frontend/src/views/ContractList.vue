<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const items = ref<any[]>([]);
onMounted(async () => {
  items.value = (await api("/contracts")).items || [];
});
</script>
<template>
  <div>
    <h2>本体契约</h2>
    <div class="kpis">
      <router-link class="card" v-for="c in items" :key="c.id" :to="`/contracts/${c.id}`">
        <div class="l">{{ c.id }}</div>
        <div class="v">{{ c.obs_dim }} / {{ c.action_dim }}</div>
        <div class="muted">{{ c.policy_hz }} Hz · {{ c.status }}</div>
      </router-link>
    </div>
  </div>
</template>
