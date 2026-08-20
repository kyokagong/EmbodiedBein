import { createRouter, createWebHistory } from "vue-router";
import Shell from "../views/Shell.vue";
import Page from "../views/Page.vue";

const p = (path: string, title: string, resource?: string) => ({
  path,
  component: Page,
  props: { title, resource: resource || path },
});

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/login", component: () => import("../views/Login.vue") },
    { path: "/showcase/:token", component: () => import("../views/Showcase.vue"), meta: { public: true } },
    {
      path: "/",
      component: Shell,
      children: [
        { path: "", component: () => import("../views/Home.vue") },
        p("/data", "数据集", "datasets"),
        { path: "/data/convert", component: Page, props: { title: "转换任务向导", resource: "jobs" } },
        { path: "/data/:id/funnel", component: () => import("../views/Funnel.vue") },
        { path: "/data/:id/episodes/:ep", component: Page, props: { title: "Episode 预览", resource: "datasets" } },
        { path: "/data/:id/slices", component: Page, props: { title: "切片管理", resource: "datasets" } },
        { path: "/data/:id", component: () => import("../views/DatasetDetail.vue") },
        p("/models", "模型", "models"),
        { path: "/models/:id/lineage", component: Page, props: { title: "血缘", resource: "models" } },
        { path: "/models/:id", component: () => import("../views/ModelDetail.vue") },
        p("/contracts", "本体契约", "contracts"),
        { path: "/contracts/:id", component: () => import("../views/ContractDetail.vue") },
        p("/assets", "机器人资产", "assets"),
        { path: "/assets/:id/viewer", component: Page, props: { title: "3D 查看器", resource: "system" } },
        { path: "/train/experiments", component: Page, props: { title: "实验", resource: "experiments" } },
        { path: "/train/submit", component: () => import("../views/Submit.vue") },
        { path: "/train/queue", component: () => import("../views/Queue.vue") },
        { path: "/train/pipelines", component: Page, props: { title: "流水线", resource: "pipelines" } },
        { path: "/train/runs/compare", component: Page, props: { title: "Run 对比", resource: "jobs" } },
        { path: "/train/runs/:id", component: () => import("../views/RunDetail.vue") },
        { path: "/eval/suites", component: Page, props: { title: "评估套件", resource: "evals/suites" } },
        { path: "/eval/reports/:id", component: Page, props: { title: "评估报告", resource: "evals/reports" } },
        { path: "/eval/compare", component: Page, props: { title: "评估对比", resource: "evals/reports" } },
        { path: "/eval/failures", component: Page, props: { title: "失败库", resource: "evals/reports" } },
        { path: "/eval/leaderboard", component: Page, props: { title: "排行榜", resource: "evals/leaderboard" } },
        { path: "/deploy/bundles", component: Page, props: { title: "策略包", resource: "bundles" } },
        { path: "/deploy/robots/:id", component: Page, props: { title: "机器人", resource: "robots" } },
        { path: "/agent", component: () => import("../views/Agent.vue") },
        { path: "/agent/audit", component: Page, props: { title: "Agent 审计", resource: "agent/audit" } },
        { path: "/settings/:rest?", component: Page, props: { title: "设置", resource: "system/version" } },
      ],
    },
  ],
});
