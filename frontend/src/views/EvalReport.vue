<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import GatePanel from "../components/GatePanel.vue";

const route = useRoute();
const row = ref<any>(null);
const drift = ref<any>(null);
const activeTab = ref("metrics");

onMounted(async () => {
  const id = route.params.id;
  row.value = await api(`/evals/reports/${id}`);
  try {
    drift.value = await api(`/evals/reports/${id}/cross-domain-drift`);
  } catch {}
});
</script>

<template>
  <div class="eval-report-page" v-if="row">
    <div class="page-head">
      <div>
        <p class="crumb"><router-link to="/eval/leaderboard">排行榜</router-link> / 评估报告 / {{ row.id }}</p>
        <h2>
          评估报告 {{ row.id }}
          <span class="badge b-cyan" v-if="row.wilson_ci">{{ row.wilson_ci.label }}</span>
        </h2>
      </div>
      <div class="row">
        <router-link to="/eval/compare" class="btn-link">📊 加入多模型对比</router-link>
      </div>
    </div>

    <!-- 4 KPI 卡片 -->
    <div class="kpis">
      <div class="card">
        <div class="l">成功率 (Wilson 95% 区间)</div>
        <div class="v" style="color: #53e5a3;">
          {{ row.metrics?.success_rate != null ? (row.metrics.success_rate * 100).toFixed(1) + '%' : '—' }}
        </div>
        <div class="muted">{{ row.wilson_ci?.label || '样本量 n=' + (row.metrics?.n_trials || 64) }}</div>
      </div>
      <div class="card">
        <div class="l">评估套件</div>
        <div class="v" style="font-size:1.1rem">{{ row.suite_id }}</div>
        <div class="muted">固定种子集测定</div>
      </div>
      <div class="card">
        <div class="l">受评模型版本</div>
        <div class="v" style="font-size:1.1rem">{{ row.model_version_id || "基线策略" }}</div>
        <div class="muted"><router-link v-if="row.model_version_id" :to="`/models/${row.model_version_id}`">查看 ModelCard</router-link><span v-else>种子基线</span></div>
      </div>
      <div class="card">
        <div class="l">sim2sim 跨域门禁</div>
        <div class="v" :style="{ color: drift?.passed_sim2sim_gate ? '#53e5a3' : '#ffc857' }">
          {{ drift ? (drift.passed_sim2sim_gate ? '✓ 通过 (≤0.6σ)' : '⚠️ 需校准') : '—' }}
        </div>
        <div class="muted">发散步数 t*={{ drift?.divergence_step ?? '—' }} 步</div>
      </div>
    </div>

    <!-- 标签页导航 -->
    <div class="tab-nav">
      <button class="tab-btn" :class="{ active: activeTab === 'metrics' }" @click="activeTab = 'metrics'">核心指标与 Seed 明细</button>
      <button class="tab-btn" :class="{ active: activeTab === 'drift' }" @click="activeTab = 'drift'">L3 跨域观测漂移热力图 (|Δμ|/σ)</button>
      <button class="tab-btn" :class="{ active: activeTab === 'gates' }" @click="activeTab = 'gates'">门禁校验结果</button>
      <button class="tab-btn" :class="{ active: activeTab === 'failures' }" @click="activeTab = 'failures'">失败案例库 (FailureCases)</button>
    </div>

    <!-- 1. 核心指标面板 -->
    <div class="panel" v-if="activeTab === 'metrics'">
      <h3>Seed 评估明细</h3>
      <table v-if="row.metrics?.per_seed && Object.keys(row.metrics.per_seed).length">
        <thead>
          <tr>
            <th>Seed</th>
            <th>成功状态</th>
            <th>Episode 步数</th>
            <th>累计回报 (Reward)</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(v, s) in row.metrics.per_seed" :key="String(s)">
            <td class="mono">seed_{{ s }}</td>
            <td><span class="badge" :class="typeof v === 'number' ? (v > 0 ? 'b-green' : 'b-amber') : (v?.success ? 'b-green' : 'b-amber')">{{ typeof v === 'number' ? (v > 0 ? '通过' : '未达成') : (v?.success ? '通过' : '未达成') }}</span></td>
            <td>{{ typeof v === 'object' ? v?.steps ?? '—' : '—' }}</td>
            <td>{{ typeof v === 'number' ? v.toFixed(2) : (v?.reward ?? '—') }}</td>
          </tr>
        </tbody>
      </table>
      <p class="muted small" v-else>无逐 Seed 明细数据；总体指标: {{ JSON.stringify(row.metrics) }}</p>
    </div>

    <!-- 2. L3 跨域漂移热力图 -->
    <div class="panel" v-if="activeTab === 'drift'">
      <h3>L3 跨域一致性诊断 (Isaac vs MuJoCo vs Real)</h3>
      <p class="muted small">{{ drift?.summary || '计算各观测项标准化分布差异 |Δμ|/σ，定位造成 sim2real gap 的物理根因项：' }}</p>
      <table v-if="drift?.terms?.length">
        <thead>
          <tr>
            <th>观测项 (Term)</th>
            <th>维度</th>
            <th>Isaac 均值</th>
            <th>MuJoCo 均值</th>
            <th>sim2sim 漂移 (|Δμ|/σ)</th>
            <th>严重度</th>
            <th>根因诊断与建议</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in drift.terms" :key="t.term">
            <td class="mono font-bold">{{ t.term }}</td>
            <td>{{ t.dim }}</td>
            <td>{{ t.isaac_mean }}</td>
            <td>{{ t.mujoco_mean }}</td>
            <td>
              <span class="drift-bar" :style="{ width: Math.min(100, t.sim2sim_drift * 120) + 'px', background: t.severity === 'critical' ? '#ff5d68' : t.severity === 'warn' ? '#ffc857' : '#53e5a3' }"></span>
              <span class="mono xs" style="margin-left:8px">{{ t.sim2sim_drift }}σ</span>
            </td>
            <td>
              <span class="badge" :class="t.severity === 'critical' ? 'b-red' : t.severity === 'warn' ? 'b-amber' : 'b-green'">
                {{ t.severity }}
              </span>
            </td>
            <td class="small">{{ t.diagnostic_hint }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 3. 门禁面板 -->
    <div class="panel" v-if="activeTab === 'gates'">
      <h3>门禁执行结论</h3>
      <GatePanel :result="row.gate_results" />
    </div>

    <!-- 4. 失败案例库 -->
    <div class="panel" v-if="activeTab === 'failures'">
      <h3>失败视频抽样与聚类</h3>
      <div class="empty-box">
        <b>本 Run 未开启相机录像或无失败视频流</b>
        仿真评测默认以无头纯计算模式运行以提高吞吐。若需录制特定失败 Episode，请在提交任务时勾选 <code>--record_video</code>。
      </div>
    </div>
  </div>
</template>

<style scoped>
.eval-report-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.tab-nav { display: flex; gap: 8px; margin: 1.5rem 0 1rem; border-bottom: 1px solid var(--line, #284052); padding-bottom: 8px; }
.tab-btn { background: transparent; border: 1px solid var(--line, #284052); color: var(--muted, #9fb4c2); padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; cursor: pointer; }
.tab-btn.active { background: rgba(83,212,255,0.15); color: var(--cyan, #53d4ff); border-color: var(--cyan); font-weight: 600; }
.drift-bar { display: inline-block; height: 8px; border-radius: 4px; vertical-align: middle; }
.font-bold { font-weight: 600; }
.btn-link { padding: 4px 12px; font-size: 0.85rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 6px; text-decoration: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.b-green { background: rgba(83, 229, 163, 0.15); color: #53e5a3; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
.b-red { background: rgba(255, 93, 104, 0.15); color: #ff5d68; }
.empty-box { background: #050c12; border: 1px dashed #284052; border-radius: 8px; padding: 1.5rem; text-align: center; color: var(--muted, #9fb4c2); line-height: 1.6; }
</style>
