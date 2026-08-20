<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const data = ref<any>(null);
onMounted(async () => {
  data.value = await api(`/contracts/${route.params.id}`);
});
</script>
<template>
  <div>
    <h2>{{ data?.id }}</h2>
    <p class="muted">obs {{ data?.obs_dim }} · act {{ data?.action_dim }} · {{ data?.policy_hz }} Hz</p>
    <p>自洽校验：<span class="badge">{{ data?.self_check?.passed ? "通过" : "失败" }}</span></p>
    <pre>{{ JSON.stringify(data?.spec, null, 2) }}</pre>
  </div>
</template>
