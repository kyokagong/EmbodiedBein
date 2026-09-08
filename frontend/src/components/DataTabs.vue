<script setup lang="ts">
import { useRoute } from "vue-router";
const props = defineProps<{ id: string }>();
const route = useRoute();
function on(kind: "overview" | "funnel" | "episodes" | "slices" | "convert") {
  const p = route.path;
  if (kind === "overview") return p === `/data/${props.id}`;
  if (kind === "funnel") return p.startsWith(`/data/${props.id}/funnel`);
  if (kind === "episodes") return p.includes("/episodes/");
  if (kind === "slices") return p.startsWith(`/data/${props.id}/slices`);
  return p === "/data/convert";
}
</script>
<template>
  <nav class="tabs">
    <router-link :to="`/data/${id}`" :class="{ on: on('overview') }">概览</router-link>
    <router-link :to="`/data/${id}/funnel`" :class="{ on: on('funnel') }">漏斗</router-link>
    <router-link :to="`/data/${id}/episodes/missing`" :class="{ on: on('episodes') }">样本</router-link>
    <router-link :to="`/data/${id}/slices`" :class="{ on: on('slices') }">切片</router-link>
    <router-link to="/data/convert" :class="{ on: on('convert') }">转换</router-link>
  </nav>
</template>
