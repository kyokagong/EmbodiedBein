<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const items = ref<any[]>([]);
onMounted(async () => {
  items.value = (await api("/embodiments")).items || [];
});
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">本体</p>
        <h2>本体 <span class="badge">{{ items.length }}</span></h2>
      </div>
      <router-link class="badge" to="/embodiments/new">登记路径</router-link>
    </div>
    <div class="grid2">
      <router-link class="card card-link" v-for="e in items" :key="e.id" :to="`/embodiments/${e.id}`">
        <div class="l mono">{{ e.id }}</div>
        <div class="v">{{ e.display_name }}</div>
        <p class="muted xs">{{ e.root_path || e.source?.path || "—" }}</p>
        <p class="xs" style="margin-top:.4rem">
          <span class="badge">{{ e.variant_count || (e.urdf_variants || []).length }} URDF</span>
          <span class="badge">{{ e.task_count || (e.gym_ids || []).length }} gym</span>
        </p>
      </router-link>
    </div>
    <p v-if="!items.length" class="muted">还没有本体。登记一份本地代码路径即可扫描 URDF。</p>
  </div>
</template>
