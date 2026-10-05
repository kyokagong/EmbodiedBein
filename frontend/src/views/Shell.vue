<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";

const route = useRoute();

const items = [
  { to: "/", label: "总览", exact: true },
  { to: "/embodiments", label: "机器人" },
  { to: "/data", label: "数据" },
  { to: "/environments", label: "环境" },
  { to: "/train", label: "训练" },
  { to: "/models", label: "模型" },
  { to: "/eval", label: "评估" },
  { to: "/deploy/bundles", label: "交付" },
];

function active(to: string, exact?: boolean) {
  if (exact) return route.path === "/";
  if (to === "/eval") return route.path === "/eval" || route.path.startsWith("/eval/");
  if (to.startsWith("/deploy/")) return route.path.startsWith("/deploy/");
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
    <main class="main">
      <nav v-if="route.path.startsWith('/data') || route.path.startsWith('/train')" class="row" aria-label="工作流导航">
        <router-link class="badge" to="/data">数据集</router-link>
        <router-link class="badge" to="/train/pipelines">流水线</router-link>
        <router-link class="badge" to="/train">训练作业</router-link>
      </nav>
      <nav v-if="route.path === '/eval' || route.path.startsWith('/eval/')" class="row" aria-label="评估导航">
        <router-link class="badge" to="/eval">评估任务</router-link>
        <router-link class="badge" to="/eval/new">新建评估</router-link>
        <router-link class="badge" to="/eval/suites">套件</router-link>
        <router-link class="badge" to="/eval/compare">对比</router-link>
        <router-link class="badge" to="/eval/failures">失败分析</router-link>
        <router-link class="badge" to="/eval/leaderboard">排行榜</router-link>
      </nav>
      <router-view />
    </main>
  </div>
</template>
