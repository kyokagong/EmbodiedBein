<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const data = ref<any>(null);
onMounted(async () => {
  data.value = await api(`/datasets/${route.params.id}`);
});
</script>
<template>
  <div>
    <h2>{{ data?.name }}</h2>
    <p class="muted">{{ data?.format }} · {{ data?.license }}</p>
    <router-link v-if="data?.id" :to="`/data/${data.id}/funnel`">漏斗审计</router-link>
    ·
    <router-link v-if="data?.id" :to="`/data/${data.id}/episodes/0`">Episode</router-link>
    ·
    <router-link v-if="data?.id" :to="`/data/${data.id}/slices`">切片</router-link>
    <table v-if="data?.versions?.length">
      <thead><tr><th>version</th><th>hash</th></tr></thead>
      <tbody>
        <tr v-for="v in data.versions" :key="v.id"><td>{{ v.version }}</td><td>{{ v.content_hash }}</td></tr>
      </tbody>
    </table>
    <p v-else class="muted">无版本</p>
  </div>
</template>
