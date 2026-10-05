<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import { VueFlow } from "@vue-flow/core";
import "@vue-flow/core/dist/style.css";
import "@vue-flow/core/dist/theme-default.css";

const props = defineProps<{ nodes: any[]; edges: any[] }>();
const router = useRouter();
const nodes = computed(() => props.nodes.map((n, i) => ({
  id: `${n.type}:${n.id}`,
  position: { x: 40 + (i % 3) * 250, y: 40 + Math.floor(i / 3) * 110 },
  data: { label: `${n.type} · ${n.name}\n${n.id}`, href: n.href },
  style: { width: "210px", whiteSpace: "pre-line", cursor: n.href ? "pointer" : "default" },
})));
const edges = computed(() => props.edges.map((e, i) => ({
  id: `e${i}`, source: `${e.src_type}:${e.src_id}`, target: `${e.dst_type}:${e.dst_id}`, label: e.relation,
})));
function openNode({ node }: { node: { data: { href?: string } } }) {
  if (node.data.href) void router.push(node.data.href);
}
</script>
<template>
  <div style="height: 360px; border: 1px solid var(--line); border-radius: 8px">
    <VueFlow :nodes="nodes" :edges="edges" fit-view-on-init @node-click="openNode" />
  </div>
  <p class="muted xs">点击节点可查看详情。</p>
  <p v-for="n in props.nodes.filter(n => n.href)" :key="`${n.type}:${n.id}`" class="xs">
    <router-link :to="n.href">{{ n.name }} · {{ n.id }}</router-link>
  </p>
</template>
