<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import GatePanel from "../components/GatePanel.vue";

const route = useRoute();
const row = ref<any>(null);
const drift = ref<any>(null);
const episodes = ref<any[]>([]);
const selectedFailure = ref<any>(null);
const activeTab = ref("metrics");
const isRpoFlat = computed(() => ["eb/rpo-flat-eval/v1", "eb/rpo-bfm-eval/v1"].includes(row.value?.gate_results?.receipt?.schema_version));
const isRpo = computed(() => ["eb/rpo-wbc-eval/v1", "eb/rpo-amp-eval/v1", "eb/rpo-flat-eval/v1", "eb/rpo-bfm-eval/v1"].includes(row.value?.gate_results?.receipt?.schema_version));
const rpoStability = computed(() => row.value?.gate_results?.success_definition === "survive_1000_steps_without_fall_or_nan");
const rpoStandWalk = computed(() => row.value?.gate_results?.receipt?.criterion === "stand_walk");
const isUfo = computed(() => row.value?.gate_results?.execution?.provider_id === "ufo");
function caseResult(ep: any) {
  if (!isRpo.value) return ep.status;
  if (ep.status === "success") return "成功";
  if (ep.fall) return `跌倒（${ep.fall_reason || '接触'}）`;
  if (ep.nan) return "数值异常";
  if (isRpoFlat.value) {
    if (ep.hard_limit) return "硬越限";
    return ep.status === "infra_failure" ? `运行异常：${ep.reason}`
      : `速度 RMSE ${ep.velocity_rmse?.toFixed(3) ?? '—'} / 0.20，转向 RMSE ${ep.yaw_rmse?.toFixed(3) ?? '—'} / 0.30`;
  }
  if (rpoStandWalk.value) return ep.scenario === "stand"
    ? `站立漂移 ${Number(ep.distance_m).toFixed(2)}m / 上限 0.5m`
    : `直行 ${Number(ep.distance_m).toFixed(2)}m / 下限 4m`;
  if (!rpoStability.value && ep.hard_limit) return "关节或力矩越限";
  if (!rpoStability.value && ["horizon", "tracking_error"].includes(ep.termination_reason)) {
    const failed = [ep.velocity_rmse > 0.15 ? `速度 RMSE ${ep.velocity_rmse.toFixed(3)} > 0.15` : "",
      ep.upper_rmse > 0.1 ? `上肢 RMSE ${ep.upper_rmse.toFixed(3)} > 0.1` : ""].filter(Boolean);
    return `走满步数，${failed.join("、") || "跟踪误差"}未达标`;
  }
  return ep.termination_reason || ep.status;
}

onMounted(async () => {
  const id = route.params.id;
  row.value = await api(`/evals/reports/${id}`);
  try { episodes.value = (await api(`/evals/reports/${id}/episodes`)).items || []; } catch {}
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

    <p v-if="isUfo" class="muted">UFO / RLinf · 源联合更新 {{ row.gate_results.source_step }} · 独立评估；缺少 case 或必需录像不能视为通过。</p>
    <!-- 4 KPI 卡片 -->
    <div class="kpis">
      <div class="card">
        <div class="l">成功率 (Wilson 95% 区间)</div>
        <div class="v" style="color: #53e5a3;">
          {{ row.metrics?.success_rate != null ? (row.metrics.success_rate * 100).toFixed(1) + '%' : '—' }}
        </div>
        <div class="muted">{{ row.wilson_ci?.label || (row.metrics?.n_trials != null ? '样本量 n=' + row.metrics.n_trials : '缺少实际样本量，未计算区间') }}</div>
      </div>
      <div class="card">
        <div class="l">评估套件</div>
        <div class="v" style="font-size:1.1rem">{{ row.suite_id }}</div>
        <div class="muted">实际配置与输入见运行记录</div>
      </div>
      <div class="card">
        <div class="l">受评模型版本</div>
        <div class="v" style="font-size:1.1rem">{{ row.model_version_id || "运行检查点" }}</div>
        <div class="muted"><router-link v-if="row.model_version_id" :to="`/models/${row.model_version_id}`">查看 ModelCard</router-link><router-link v-else-if="row.run_id" :to="`/eval/runs/${row.run_id}`">查看评估 Run</router-link><span v-else>未记录模型版本</span></div>
        <div v-if="row.gate_results?.checkpoint?.run_id" class="muted xs">源 checkpoint：{{ row.gate_results.checkpoint.run_id }} / step {{ row.gate_results.checkpoint.step }}</div>
      </div>
      <div class="card" v-if="!row.metrics?.profile && !isRpo">
        <div class="l">sim2sim 跨域门禁</div>
        <div class="v" :style="{ color: drift?.passed_sim2sim_gate ? '#53e5a3' : '#ffc857' }">
          {{ drift?.passed_sim2sim_gate == null ? '尚未评估' : (drift.passed_sim2sim_gate ? '✓ 通过' : '⚠️ 未通过') }}
        </div>
        <div class="muted">发散步数 t*={{ drift?.divergence_step ?? '—' }} 步</div>
      </div>
    </div>

    <!-- 标签页导航 -->
    <div class="tab-nav">
      <button class="tab-btn" :class="{ active: activeTab === 'metrics' }" @click="activeTab = 'metrics'">{{ row.metrics?.profile ? '任务与案例结果' : '核心指标与 Seed 明细' }}</button>
      <button v-if="!row.metrics?.profile && !isRpo" class="tab-btn" :class="{ active: activeTab === 'drift' }" @click="activeTab = 'drift'">L3 跨域观测漂移热力图 (|Δμ|/σ)</button>
      <button class="tab-btn" :class="{ active: activeTab === 'gates' }" @click="activeTab = 'gates'">门禁校验结果</button>
      <button class="tab-btn" :class="{ active: activeTab === 'failures' }" @click="activeTab = 'failures'">失败案例库 (FailureCases)</button>
    </div>

    <!-- 1. 核心指标面板 -->
    <div class="panel" v-if="activeTab === 'metrics'">
      <p>证据：{{ row.gate_results?.validity || '历史报告，未验证逐案例完整性' }} · 资格：{{ row.gate_results?.qualification || '未判定' }}</p>
      <p v-if="row.metrics?.planned != null">已完成 {{ row.metrics.completed }} / 计划 {{ row.metrics.planned }} · 成功 {{ row.metrics.successes }}</p>
      <p v-if="isRpoFlat">官方完整速度/转向指令，每例 20 秒；无跌倒、NaN 或硬越限，平移向量 RMSE ≤0.20 m/s、转向 RMSE ≤0.30 rad/s。实际成功 {{ row.metrics?.successes }} / {{ row.metrics?.n_trials }}；最终验收为 100 例至少 80 例通过。</p>
      <p v-else-if="isRpo">成功标准：{{ rpoStandWalk ? '平地直立或 0.3m/s 直行：保持直立 1000 步，站立漂移 ≤ 0.5m，直行前进 ≥ 4m' : rpoStability ? '身体稳定：达到 1000 步且未跌倒、无数值异常' : '跟踪精度：无跌倒、越限或数值异常，且速度 RMSE ≤ 0.15、上肢 RMSE ≤ 0.1' }}</p>
      <p v-if="isRpo">环境 {{ row.gate_results?.receipt?.backend === 'isaac' ? 'Isaac Lab' : 'MuJoCo' }} · 跌倒 {{ row.metrics?.fall_count }} · 关节/力矩越限 {{ row.metrics?.hard_limit_count }} · 数值异常 {{ row.metrics?.nan_count }} · 速度 RMSE {{ row.metrics?.velocity_rmse }} · {{ isRpoFlat ? '转向' : '上肢' }} RMSE {{ isRpoFlat ? row.metrics?.yaw_rmse : row.metrics?.upper_rmse }}</p>
      <h3 v-if="row.metrics?.per_task">逐任务结果</h3>
      <table v-if="row.metrics?.per_task"><thead><tr><th>Task</th><th>成功</th><th>有效次数</th><th>成功率</th></tr></thead>
        <tbody><tr v-for="(v, task) in row.metrics.per_task" :key="String(task)"><td>task {{ task }}</td>
          <td>{{ v.successes }}</td><td>{{ v.n_trials }}</td><td>{{ (100 * v.successes / v.n_trials).toFixed(1) }}%</td></tr></tbody></table>
      <h3 v-if="episodes.length">实际案例</h3>
      <table v-if="episodes.length"><thead><tr><th>Case</th><th>任务</th><th>初始状态</th><th v-if="rpoStandWalk">场景</th><th>结果</th><th>步数</th><th v-if="rpoStandWalk">位移</th></tr></thead>
        <tbody><tr v-for="ep in episodes" :key="ep.case_id"><td class="mono xs">{{ ep.case_id }}</td><td>{{ ep.task_id }}</td>
          <td>{{ ep.init_state_id ?? ep.seed }}</td><td v-if="rpoStandWalk">{{ ep.scenario === 'stand' ? '原地站立' : '直线行走' }}</td><td>{{ caseResult(ep) }} <button v-if="ep.video_rel" type="button" @click="selectedFailure = ep">播放录像</button></td><td>{{ ep.episode_len ?? '—' }}</td><td v-if="rpoStandWalk">{{ Number(ep.distance_m).toFixed(2) }}m</td></tr></tbody></table>
      <div v-if="selectedFailure?.video_rel" class="panel"><h3>{{ selectedFailure.case_id }} · 录像</h3>
        <video controls preload="metadata" style="width:100%;max-width:720px" :src="`/api/v1/evals/reports/${row.id}/media/${selectedFailure.video_rel.replace('seed_','').replace('.mp4','')}`">浏览器无法播放该录像。</video></div>
      <p v-else-if="!episodes.length" class="muted">此报告没有逐案例索引；历史汇总结果不能用于配对比较。</p>
      <h3 v-if="!row.metrics?.profile && !isRpo">Seed 评估明细</h3>
      <table v-if="!row.metrics?.profile && !isRpo && row.metrics?.per_seed && Object.keys(row.metrics.per_seed).length">
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
      <p class="muted small" v-else-if="!row.metrics?.profile && !isRpo">无逐 Seed 明细数据；总体指标: {{ JSON.stringify(row.metrics) }}</p>
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
      <h3>失败案例与录像</h3>
      <table v-if="episodes.some(ep => ['task_failure','timeout','invalid'].includes(ep.status))"><thead><tr><th>Case</th><th>任务</th><th>初始状态</th><th>原因</th></tr></thead>
        <tbody><tr v-for="ep in episodes.filter(ep => ['task_failure','timeout','invalid'].includes(ep.status))" :key="ep.case_id"><td class="mono xs">{{ ep.case_id }}</td><td>{{ ep.task_id }}</td><td>{{ ep.init_state_id }}</td><td>{{ caseResult(ep) }} <button v-if="ep.video_rel" type="button" @click="selectedFailure = ep">查看录像</button><span v-else-if="ep.video_status === 'unavailable'"> · 录像不可用</span></td></tr></tbody></table>
      <div v-if="selectedFailure?.video_rel" class="panel"><h3>{{ selectedFailure.case_id }} · 运行录像</h3>
        <video controls preload="metadata" style="width:100%;max-width:720px" :src="`/api/v1/evals/reports/${row.id}/media/${selectedFailure.video_rel.replace('seed_','').replace('.mp4','')}`">浏览器无法播放该录像。</video>
        <p v-if="!isRpo" class="muted">录像可能包含同一环境批次的其他案例；按 case ID 和初始状态核对。</p></div>
      <div v-else class="empty-box">
        <b>尚无已索引的失败视频</b>
        是否生成录像以本次运行配置与产物为准；当前报告尚未提供失败片段索引。
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
