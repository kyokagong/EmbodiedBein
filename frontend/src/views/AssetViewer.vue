<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import EmptyState from "../components/EmptyState.vue";

const route = useRoute();
const asset = ref<any>(null);
const el = ref<HTMLDivElement | null>(null);
const note = ref("无网格文件，不渲染假模型。");

onMounted(async () => {
  const items = (await api("/assets")).items || [];
  asset.value = items.find((x: any) => x.id === route.params.id || x.contract_id === route.params.id) || items[0];
  const path = asset.value?.mjcf || asset.value?.usd || "";
  if (!path) return;
  note.value = `已登记路径 ${path}。three.js 仅在浏览器能取到网格时加载。`;
  try {
    const THREE = await import("three");
    if (!el.value) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
    camera.position.set(1.2, 1.2, 1.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(el.value.clientWidth || 640, 280);
    el.value.appendChild(renderer.domElement);
    scene.add(new THREE.AxesHelper(0.5));
    renderer.render(scene, camera);
  } catch {
    note.value = `three.js 未加载。路径：${path}`;
  }
});
</script>
<template>
  <div>
    <h2>3D 查看器 {{ route.params.id }}</h2>
    <EmptyState v-if="!asset" title="缺资产">没有 URDF/MJCF 登记。</EmptyState>
    <p v-else class="muted">{{ note }}</p>
    <div ref="el" />
  </div>
</template>
