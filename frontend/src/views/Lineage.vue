<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const edges = ref<any[]>([]);
onMounted(async () => {
  edges.value = (await api(`/models/${route.params.id}/lineage`)).edges || [];
});
</script>
<template>
  <div>
    <h2>血缘</h2>
    <ul>
      <li v-for="(e,i) in edges" :key="i">{{ e.src_type }}:{{ e.src_id }} —{{ e.relation }}→ {{ e.dst_type }}:{{ e.dst_id }}</li>
    </ul>
    <p class="muted" v-if="!edges.length">暂无血缘边</p>
  </div>
</template>
