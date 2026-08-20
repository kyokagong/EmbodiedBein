<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const ver = ref<any>(null);
onMounted(async () => { ver.value = await api("/system/version"); });
</script>
<template>
  <div>
    <h2>设置</h2>
    <p>版本 {{ ver?.version }} · {{ ver?.api }}</p>
    <p class="muted">单用户、无 OAuth。Redis 6380。</p>
  </div>
</template>
