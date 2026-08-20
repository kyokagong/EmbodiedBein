<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const title = defineProps<{ title: string; resource: string }>();
const data = ref<any>(null);
const err = ref("");
onMounted(async () => {
  try {
    const path = title.resource.startsWith("/") ? title.resource : `/${title.resource}`;
    data.value = await api(path);
  } catch (e: any) {
    err.value = e.message;
  }
});
</script>
<template>
  <div>
    <h2>{{ title }}</h2>
    <p v-if="err" class="error">{{ err }}</p>
    <pre v-else>{{ JSON.stringify(data, null, 2) }}</pre>
  </div>
</template>
