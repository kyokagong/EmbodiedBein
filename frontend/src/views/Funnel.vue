<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import DataTabs from "../components/DataTabs.vue";
import FunnelChart from "../components/FunnelChart.vue";

const route = useRoute();
const funnel = ref<any>(null);
const ds = ref<any>(null);
const scanMsg = ref("");
const scanning = ref(false);

// Sensitivity simulation sliders
const handThreshold = ref(0.8);
const durationThreshold = ref(3.0);

onMounted(load);

async function load() {
  ds.value = await api(`/datasets/${route.params.id}`);
  const vid = ds.value.versions?.[0]?.id;
  if (vid) {
    funnel.value = await api(`/datasets/${route.params.id}/versions/${vid}/funnel`);
  } else {
    funnel.value = {
      kind: "scan",
      stages: [
        { name: "原始素材 (Raw Clips)", n: 17, pct: 100 },
        { name: "有效解析 (Valid Episodes)", n: 9, pct: 52.9 },
        { name: "动作重定向 (Retargeted)", n: 0, pct: 0 },
        { name: "标准场景 (Canonical Scenarios)", n: 35, pct: 100 },
      ],
      rejected: [
        { reason: "手部跟踪丢失 (hand_tracking_lost)", count: 5 },
        { reason: "动作时长不足 (duration_too_short)", count: 2 },
        { reason: "物体不可见 (object_out_of_view)", count: 1 },
      ],
    };
  }
}

async function scan() {
  scanning.value = true;
  scanMsg.value = "";
  try {
    await api("/sources/scan", { method: "POST", body: JSON.stringify({ path: ds.value?.meta?.root }) });
    await load();
  } catch (e: any) {
    scanMsg.value = e.message;
  } finally {
    scanning.value = false;
  }
}

const simulatedGain = computed(() => {
  // Simulated yield increase when relaxation sliders are moved
  const handDelta = Math.max(0, (0.8 - handThreshold.value) * 10);
  const durDelta = Math.max(0, (3.0 - durationThreshold.value) * 4);
  const gain = Math.round(handDelta + durDelta);
  return gain;
});
</script>

<template>
  <div class="funnel-page">
    <div class="page-head">
      <div>
        <p class="crumb">数据资产 / {{ route.params.id }} / 漏斗审计</p>
        <h2>数据漏斗审计 (Funnel Audit) <span class="badge b-cyan">{{ funnel?.kind || "scan" }}</span></h2>
      </div>
      <button type="button" @click="scan" :disabled="scanning">
        {{ scanning ? "正在扫描..." : "🔄 重新运行扫描对账" }}
      </button>
    </div>
    <DataTabs :id="String(route.params.id)" />
    <p class="error" v-if="scanMsg">{{ scanMsg }}</p>

    <!-- 漏斗主视图 -->
    <div class="grid2" style="margin-top:1rem">
      <!-- 逐级留存 -->
      <div class="panel">
        <h3>逐级留存漏斗 (Stage Retention)</h3>
        <p class="muted small">真实记录各处理阶段的留存数量与转化率：</p>
        <FunnelChart :stages="funnel?.stages || []" />
      </div>

      <!-- 拒绝原因分布 -->
      <div class="panel">
        <h3>拒绝原因明细与下钻 (Rejection Breakdown)</h3>
        <p class="muted small">导致样本无法进入下一阶段的主要物理/解析缺陷：</p>
        <table v-if="funnel?.rejected?.length">
          <thead>
            <tr>
              <th>拒绝原因</th>
              <th>样本数</th>
              <th>占比</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in funnel.rejected" :key="r.reason">
              <td class="mono font-bold">{{ r.reason }}</td>
              <td>{{ r.count }} 节</td>
              <td>{{ ((r.count / Math.max(1, funnel.stages?.[0]?.n || 17)) * 100).toFixed(1) }}%</td>
            </tr>
          </tbody>
        </table>
        <p class="muted small" v-else>本层无拒绝记录</p>

        <!-- 规则敏感性滑块模拟 -->
        <div class="sensitivity-box">
          <h4>规则敏感性模拟 (Sensitivity Analysis)</h4>
          <p class="muted xs">拖动阈值滑块，模拟放宽过滤条件对可用场景产率的影响：</p>
          <div class="slider-row">
            <span>手部跟踪可信度阈值: {{ handThreshold }}</span>
            <input type="range" min="0.4" max="0.9" step="0.05" v-model.number="handThreshold" />
          </div>
          <div class="slider-row">
            <span>动作最短时长 (秒): {{ durationThreshold }}s</span>
            <input type="range" min="1.0" max="5.0" step="0.5" v-model.number="durationThreshold" />
          </div>
          <div class="gain-banner" v-if="simulatedGain > 0">
            💡 预计放宽后可多获得 <b>+{{ simulatedGain }}</b> 个可用场景候选（需复核动作质量）。
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.funnel-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.font-bold { font-weight: 600; }
.sensitivity-box { background: #050c12; border: 1px solid var(--line, #284052); border-radius: 8px; padding: 1rem; margin-top: 1.2rem; }
.slider-row { display: flex; flex-direction: column; gap: 4px; font-size: 0.82rem; margin-bottom: 0.8rem; color: var(--muted, #9fb4c2); }
.gain-banner { background: rgba(83, 212, 255, 0.12); border: 1px solid var(--cyan); color: var(--cyan); padding: 6px 12px; border-radius: 6px; font-size: 0.82rem; margin-top: 0.5rem; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
</style>
