import { createRouter, createWebHistory } from "vue-router";
import Shell from "../views/Shell.vue";
import { api } from "../api/client";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/login", component: () => import("../views/Login.vue") },
    { path: "/showcase/:token", component: () => import("../views/Showcase.vue"), meta: { public: true } },
    {
      path: "/",
      component: Shell,
      children: [
        { path: "", component: () => import("../views/Home.vue") },
        { path: "/data/new", component: () => import("../views/DatasetNew.vue") },
        { path: "/data", component: () => import("../views/DatasetList.vue") },
        { path: "/data/convert", component: () => import("../views/DataConvert.vue") },
        { path: "/data/:id/funnel", component: () => import("../views/Funnel.vue") },
        { path: "/data/:id/episodes/:ep", component: () => import("../views/EpisodePreview.vue") },
        { path: "/data/:id/slices", component: () => import("../views/DatasetSlices.vue") },
        { path: "/data/:id", component: () => import("../views/DatasetDetail.vue") },
        { path: "/models", component: () => import("../views/ModelList.vue") },
        { path: "/models/:id/lineage", component: () => import("../views/Lineage.vue") },
        { path: "/models/:id", component: () => import("../views/ModelDetail.vue") },
        { path: "/embodiments", component: () => import("../views/EmbodimentList.vue") },
        { path: "/embodiments/new", component: () => import("../views/EmbodimentNew.vue") },
        { path: "/embodiments/:id/envs/:eeId", component: () => import("../views/EnvDetail.vue") },
        { path: "/embodiments/:id", component: () => import("../views/EmbodimentDetail.vue") },
        { path: "/tasks", component: () => import("../views/TaskList.vue") },
        { path: "/assets", component: () => import("../views/AssetList.vue") },
        { path: "/assets/:id/viewer", component: () => import("../views/AssetViewer.vue") },
        { path: "/envs", component: () => import("../views/EnvHub.vue") },
        { path: "/environments", component: () => import("../views/EnvironmentCatalog.vue") },
        { path: "/environments/:id", component: () => import("../views/EnvironmentCatalog.vue") },
        { path: "/train", component: () => import("../views/TrainList.vue") },
        { path: "/train/new", component: () => import("../views/TrainNew.vue") },
        { path: "/train/experiments", component: () => import("../views/Experiments.vue") },
        { path: "/train/submit", component: () => import("../views/Submit.vue") },
        { path: "/train/queue", component: () => import("../views/Queue.vue") },
        { path: "/train/pipelines", component: () => import("../views/Pipelines.vue") },
        { path: "/train/runs/compare", component: () => import("../views/RunCompare.vue") },
        { path: "/train/runs/:id", component: () => import("../views/RunDetail.vue") },
        { path: "/eval/runs/:id", component: () => import("../views/RunDetail.vue") },
        { path: "/train/:id", component: () => import("../views/TrainJobDetail.vue") },
        { path: "/eval", component: () => import("../views/EvalJobs.vue") },
        { path: "/eval/new", component: () => import("../views/EvalNew.vue") },
        { path: "/eval/jobs/:id", component: () => import("../views/EvalJobDetail.vue") },
        { path: "/eval/suites", component: () => import("../views/EvalSuites.vue") },
        { path: "/eval/reports/:id", component: () => import("../views/EvalReport.vue") },
        { path: "/eval/compare", component: () => import("../views/EvalCompare.vue") },
        { path: "/eval/failures", component: () => import("../views/EvalFailures.vue") },
        { path: "/eval/leaderboard", component: () => import("../views/Leaderboard.vue") },
        { path: "/deploy/bundles", component: () => import("../views/Bundles.vue") },
        { path: "/deploy/simulation", component: () => import("../views/SimulationValidation.vue") },
        { path: "/deploy/robots/:id", component: () => import("../views/RobotDetail.vue") },
        { path: "/agent", component: () => import("../views/Agent.vue") },
        { path: "/agent/audit", component: () => import("../views/AgentAudit.vue") },
        { path: "/settings/:rest?", component: () => import("../views/Settings.vue") },
      ],
    },
  ],
});

router.beforeEach(async (to) => {
  if (to.path === "/login" || to.meta.public) return true;
  try {
    await api("/auth/me");
    return true;
  } catch {
    return "/login";
  }
});

export default router;
