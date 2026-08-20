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
    <pre>{{ JSON.stringify(data?.versions, null, 2) }}</pre>
  </div>
</template>
