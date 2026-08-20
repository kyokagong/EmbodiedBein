<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
const route = useRoute();
const asset = ref<any>(null);
onMounted(async () => {
  const items = (await api("/assets")).items || [];
  asset.value = items.find((x: any) => x.contract_id === route.params.id) || items[0];
});
</script>
<template>
  <div>
    <h2>3D 查看器 {{ route.params.id }}</h2>
    <p class="muted">URDF/MJCF 懒加载。无文件时显示路径而不是假模型。</p>
    <p v-if="asset">MJCF {{ asset.mjcf || "无" }} · USD {{ asset.usd || "无" }}</p>
    <p class="muted">three.js 将在存在网格文件时加载；当前资产仅登记路径。</p>
  </div>
</template>
