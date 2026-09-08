<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";

const route = useRoute();

const items = [
  { to: "/", label: "总览", exact: true },
  { to: "/embodiments", label: "本体" },
  { to: "/data", label: "数据" },
  { to: "/envs", label: "环境" },
  { to: "/train", label: "训练" },
  { to: "/models", label: "模型" },
];

function active(to: string, exact?: boolean) {
  if (exact) return route.path === "/";
  return route.path === to || route.path.startsWith(to + "/");
}

const crumb = computed(() => route.path);
</script>
<template>
  <div class="shell">
    <nav class="nav">
      <div>
        <h1>EmbodiedBein</h1>
        <p class="brand-sub">RLinf 具身训练工作台</p>
        <router-link
          v-for="it in items"
          :key="it.to"
          :to="it.to"
          active-class="nav-noop"
          exact-active-class="nav-noop"
          :class="{ 'router-link-active': active(it.to, it.exact) }"
        >{{ it.label }}</router-link>
      </div>
      <div class="nav-foot">
        <router-link
          to="/settings"
          active-class="nav-noop"
          exact-active-class="nav-noop"
          :class="{ 'router-link-active': crumb.startsWith('/settings') }"
        >设置</router-link>
      </div>
    </nav>
    <main class="main"><router-view /></main>
  </div>
</template>
