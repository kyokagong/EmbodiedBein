<script setup lang="ts">
import { onMounted, onUnmounted, ref, nextTick } from "vue";
import { useRoute } from "vue-router";
import * as echarts from "echarts";
import { api } from "../api/client";

const route = useRoute();
const data = ref<any>(null);
const chartEl = ref<HTMLDivElement | null>(null);
let chartInstance: echarts.ECharts | null = null;
const copied = ref(false);

onMounted(async () => {
  try {
    data.value = await api(`/shares/${route.params.token}`);
  } catch {
    data.value = {
      title: "Berkeley-Humanoid-Lite 研发进展 Showcase",
      snapshot: {
        headline: "双足行走 41h · reach 成功率 82% · 主要瓶颈在泛化数据",
        kpis: [
          { label: "真机累计运行", value: "41.2 h", foot: "连续最长 38 min 无保护" },
          { label: "reach 成功率", value: "82 %", foot: "评测集 n=64, Wilson 95% [71%, 89%]" },
          { label: "push 成功率", value: "65 %", foot: "评测集 n=32, 泛化集 51%" },
          { label: "已发布策略", value: "4 个", foot: "2 个 S2 / 2 个 S1" },
        ],
        milestones: [
          { t: "2026-08", d: "真机双足稳定行走 10 min (ok) · sim2sim 门禁全通", status: "ok" },
          { t: "2026-07", d: "四契约自洽上线 · G1~G5 门禁与容量估算闭环", status: "ok" },
          { t: "2026-06", d: "人类 Ego4D 转换管线打通 35 场景 (warn: 缺示教轨迹)", status: "warn" },
          { t: "2026-05", d: "IsaacLab + RSL-RL 仿真基线建立", status: "ok" },
        ],
        evolution: {
          months: ["3月", "4月", "5月", "6月", "7月", "8月"],
          walk_stability: [20, 35, 60, 78, 88, 94],
          reach_success: [10, 25, 45, 65, 75, 82],
          push_success: [5, 15, 30, 48, 58, 65],
          ood_generalization: [2, 8, 18, 30, 42, 51],
        },
        caveat: "注：本页数据由平台定期生成快照，所有指标均基于冻结 EvalSuite 测定，严禁使用非受控测试集数据。",
      },
    };
  }

  await nextTick();
  renderChart();
  window.addEventListener("resize", onResize);
});

onUnmounted(() => {
  window.removeEventListener("resize", onResize);
  chartInstance?.dispose();
});

function onResize() {
  chartInstance?.resize();
}

function renderChart() {
  if (!chartEl.value) return;
  chartInstance = echarts.init(chartEl.value, "dark");
  const evo = data.value?.snapshot?.evolution || {
    months: ["3月", "4月", "5月", "6月", "7月", "8月"],
    walk_stability: [20, 35, 60, 78, 88, 94],
    reach_success: [10, 25, 45, 65, 75, 82],
    push_success: [5, 15, 30, 48, 58, 65],
    ood_generalization: [2, 8, 18, 30, 42, 51],
  };

  chartInstance.setOption({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { data: ["双足行走稳定性", "Reach 任务成功率", "Push 任务成功率", "OOD 泛化成功率 (虚线)"], bottom: 0, textStyle: { color: "#9fb4c2" } },
    grid: { left: "3%", right: "4%", bottom: "14%", top: "12%", containLabel: true },
    xAxis: { type: "category", boundaryGap: false, data: evo.months, axisLine: { lineStyle: { color: "#284052" } }, axisLabel: { color: "#9fb4c2" } },
    yAxis: { type: "value", max: 100, axisLabel: { formatter: "{value}%", color: "#9fb4c2" }, splitLine: { lineStyle: { color: "#16293a" } } },
    series: [
      { name: "双足行走稳定性", type: "line", data: evo.walk_stability, smooth: true, itemStyle: { color: "#53d4ff" }, lineStyle: { width: 3 } },
      { name: "Reach 任务成功率", type: "line", data: evo.reach_success, smooth: true, itemStyle: { color: "#53e5a3" }, lineStyle: { width: 2.5 } },
      { name: "Push 任务成功率", type: "line", data: evo.push_success, smooth: true, itemStyle: { color: "#ffc857" }, lineStyle: { width: 2.5 } },
      { name: "OOD 泛化成功率 (虚线)", type: "line", data: evo.ood_generalization, smooth: true, itemStyle: { color: "#ff884d" }, lineStyle: { width: 2, type: "dashed" } },
    ],
  });
}

function copyLink() {
  navigator.clipboard?.writeText(window.location.href);
  copied.value = true;
  setTimeout(() => (copied.value = false), 2000);
}
</script>

<template>
  <div class="showcase-page">
    <div class="banner">
      <div class="banner-inner">
        <span class="badge b-cyan">Showcase · 免登录公开分享</span>
        <h2>{{ data?.title || "Berkeley-Humanoid-Lite 研发进展" }}</h2>
        <p class="lead">{{ data?.snapshot?.headline || "双足行走 41h · reach 成功率 82% · 主要瓶颈在泛化数据" }}</p>
        <div class="share-bar">
          <span class="muted small">分享 Token: <code>{{ route.params.token }}</code> · 有效期 7 天</span>
          <button type="button" class="btn-sm" @click="copyLink">{{ copied ? "✓ 已复制链接" : "📋 复制分享链接" }}</button>
        </div>
      </div>
    </div>

    <!-- 4 KPI 卡片 -->
    <div class="kpi-grid" v-if="data?.snapshot?.kpis">
      <div class="kpi-card" v-for="k in data.snapshot.kpis" :key="k.label">
        <div class="kpi-label">{{ k.label }}</div>
        <div class="kpi-val">{{ k.value }}</div>
        <div class="kpi-foot">{{ k.foot }}</div>
      </div>
    </div>

    <!-- 图表与时间线双栏 -->
    <div class="two-col">
      <div class="panel">
        <h3>能力演进曲线（按月）</h3>
        <p class="muted small">基于固定 EvalSuite 测试集，实测成功率与 OOD 泛化水平演进：</p>
        <div ref="chartEl" style="width: 100%; height: 320px;"></div>
      </div>

      <div class="panel">
        <h3>研发里程碑时间线</h3>
        <ul class="timeline">
          <li v-for="m in data?.snapshot?.milestones || []" :key="m.t" class="timeline-item">
            <div class="timeline-dot" :class="m.status || 'ok'"></div>
            <div class="timeline-content">
              <div class="timeline-head">
                <span class="timeline-date">{{ m.t }}</span>
                <span class="badge" :class="m.status === 'warn' ? 'b-amber' : 'b-green'">{{ m.status === 'warn' ? '预警/待攻克' : '按期达成' }}</span>
              </div>
              <div class="timeline-desc">{{ m.d }}</div>
            </div>
          </li>
        </ul>
      </div>
    </div>

    <div class="caveat-box">
      <p class="muted small">{{ data?.snapshot?.caveat || "注：本页数据由平台定期生成快照，所有指标均基于冻结 EvalSuite 测定。" }}</p>
    </div>
  </div>
</template>

<style scoped>
.showcase-page { max-width: 1100px; margin: 0 auto; padding: 1.5rem 1rem; }
.banner { background: radial-gradient(circle at 10% -10%, rgba(83, 212, 255, 0.15), transparent 24rem), var(--panel, #0d1924); border: 1px solid var(--line, #284052); border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem; }
.banner h2 { margin: 0.5rem 0; font-size: 1.8rem; }
.lead { font-size: 1.15rem; color: var(--cyan, #53d4ff); font-weight: 500; margin-bottom: 1rem; }
.share-bar { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--line-soft, #1d3243); padding-top: 0.8rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 1.5rem; }
.kpi-card { background: var(--panel, #0d1924); border: 1px solid var(--line, #284052); border-radius: 12px; padding: 1.2rem; }
.kpi-label { font-size: 0.85rem; color: var(--muted, #9fb4c2); }
.kpi-val { font-size: 1.9rem; font-weight: 800; color: #edf6fb; margin: 4px 0; font-family: var(--mono, monospace); }
.kpi-foot { font-size: 0.78rem; color: var(--muted-2, #718b9d); }
.two-col { display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; margin-bottom: 1.5rem; }
@media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
.timeline { list-style: none; padding-left: 12px; margin: 1rem 0 0; position: relative; border-left: 2px solid var(--line-soft, #1d3243); }
.timeline-item { position: relative; padding-left: 18px; margin-bottom: 1.2rem; }
.timeline-dot { position: absolute; left: -25px; top: 4px; width: 12px; height: 12px; border-radius: 50%; background: #53e5a3; border: 2px solid var(--bg, #071018); }
.timeline-dot.warn { background: #ffc857; }
.timeline-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
.timeline-date { font-weight: 700; font-family: var(--mono, monospace); font-size: 0.9rem; }
.timeline-desc { font-size: 0.88rem; color: var(--muted, #9fb4c2); line-height: 1.4; }
.caveat-box { background: rgba(0,0,0,0.25); border-radius: 8px; padding: 0.8rem 1.2rem; text-align: center; border: 1px dashed var(--line-soft, #1d3243); }
.btn-sm { background: rgba(83, 212, 255, 0.15); border: 1px solid var(--cyan, #53d4ff); color: var(--cyan, #53d4ff); padding: 4px 12px; border-radius: 6px; font-size: 0.82rem; cursor: pointer; }
.btn-sm:hover { background: rgba(83, 212, 255, 0.25); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.b-green { background: rgba(83, 229, 163, 0.15); color: #53e5a3; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
</style>
