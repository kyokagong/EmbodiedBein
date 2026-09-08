<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "../api/client";

const reports = ref<any[]>([]);
const selectedIds = ref<string[]>([]);

onMounted(async () => {
  reports.value = (await api("/evals/reports")).items || [];
  if (reports.value.length >= 2) {
    selectedIds.value = [reports.value[0].id, reports.value[1].id];
  } else if (reports.value.length === 1) {
    selectedIds.value = [reports.value[0].id];
  }
});

const selectedReports = computed(() => {
  return reports.value.filter((r) => selectedIds.value.includes(r.id));
});

const significanceVerdict = computed(() => {
  if (selectedReports.value.length < 2) return null;
  const a = selectedReports.value[0];
  const b = selectedReports.value[1];
  const ciA = a.wilson_ci;
  const ciB = b.wilson_ci;
  if (!ciA || !ciB) return { significant: false, text: "样本量不足，无法判定显著性" };
  const overlap = !(ciA.lower > ciB.upper || ciB.lower > ciA.upper);
  if (overlap) {
    return {
      significant: false,
      text: `差异不显著（Wilson 95% 区间重叠：[${(ciA.lower * 100).toFixed(1)}%, ${(ciA.upper * 100).toFixed(1)}%] ∩ [${(ciB.lower * 100).toFixed(1)}%, ${(ciB.upper * 100).toFixed(1)}%]）`,
    };
  }
  return {
    significant: true,
    text: `差异显著（p < 0.05，置信区间完全脱离：[${(ciA.lower * 100).toFixed(1)}%, ${(ciA.upper * 100).toFixed(1)}%] vs [${(ciB.lower * 100).toFixed(1)}%, ${(ciB.upper * 100).toFixed(1)}%]）`,
  };
});

function toggleSelect(id: string) {
  if (selectedIds.value.includes(id)) {
    selectedIds.value = selectedIds.value.filter((x) => x !== id);
  } else {
    if (selectedIds.value.length >= 4) selectedIds.value.shift();
    selectedIds.value.push(id);
  }
}
</script>

<template>
  <div class="eval-compare-page">
    <div class="page-head">
      <div>
        <p class="crumb">评估与基准 / 策略对比</p>
        <h2>策略评估配对对比 (Eval Compare)</h2>
      </div>
      <router-link to="/eval/leaderboard" class="btn-link">🏆 查看排行榜</router-link>
    </div>
    <p class="muted small">选择 2~4 个评估报告进行并列对比。系统将基于 Wilson 95% 置信区间自动给出差异显著性判定，避免因偶然波动下结论。</p>

    <!-- 显著性判定提示卡 -->
    <div class="card" v-if="significanceVerdict" :style="{ borderColor: significanceVerdict.significant ? 'var(--green, #53e5a3)' : 'var(--amber, #ffc857)' }">
      <div class="l">统计学显著性结论</div>
      <div class="v" :style="{ color: significanceVerdict.significant ? '#53e5a3' : '#ffc857' }">
        {{ significanceVerdict.significant ? '✓ 策略表现存在显著差异' : '⚠️ 差异不显著 (No Statistically Significant Difference)' }}
      </div>
      <p class="muted small mb0">{{ significanceVerdict.text }}</p>
    </div>

    <!-- 报告勾选列表 -->
    <div class="panel" style="margin-top:1rem">
      <h3>选择要对比的评估报告 (最多 4 项)</h3>
      <div class="report-chips">
        <label v-for="r in reports" :key="r.id" class="chip-item" :class="{ selected: selectedIds.includes(r.id) }">
          <input type="checkbox" :checked="selectedIds.includes(r.id)" @change="toggleSelect(r.id)" />
          <span class="mono">{{ r.id }}</span>
          <span class="muted small">({{ r.model_version_id || '基线' }} · {{ r.wilson_ci?.label || (r.metrics?.success_rate ? (r.metrics.success_rate*100).toFixed(0)+'%' : '—') }})</span>
        </label>
      </div>
    </div>

    <!-- 并列对比矩阵 -->
    <div class="panel" v-if="selectedReports.length" style="margin-top:1rem; overflow-x:auto">
      <h3>并列对比矩阵</h3>
      <table>
        <thead>
          <tr>
            <th style="min-width:160px">对比维度</th>
            <th v-for="r in selectedReports" :key="r.id" style="min-width:200px">
              <div class="mono font-bold">{{ r.id }}</div>
              <div class="small muted">{{ r.model_version_id || '基线策略' }}</div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>评估套件 (Suite)</td>
            <td v-for="r in selectedReports" :key="r.id" class="mono small">{{ r.suite_id }}</td>
          </tr>
          <tr>
            <td>成功率 (Wilson 95% CI)</td>
            <td v-for="r in selectedReports" :key="r.id">
              <span class="mono font-bold" style="color:#53e5a3">
                {{ r.wilson_ci?.label || ((r.metrics?.success_rate != null ? (r.metrics.success_rate * 100).toFixed(1) + '%' : '—')) }}
              </span>
            </td>
          </tr>
          <tr>
            <td>平均回报 (Mean Reward)</td>
            <td v-for="r in selectedReports" :key="r.id" class="mono font-bold">
              {{ r.metrics?.mean_reward != null ? r.metrics.mean_reward.toFixed(2) : '—' }}
            </td>
          </tr>
          <tr>
            <td>门禁断言结论</td>
            <td v-for="r in selectedReports" :key="r.id">
              <span class="badge" :class="(r.gate_results?.passed ?? true) ? 'b-green' : 'b-amber'">
                {{ (r.gate_results?.passed ?? true) ? '✓ 门禁通过' : '⚠️ 门禁告警' }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.eval-compare-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.report-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
.chip-item { display: flex; align-items: center; gap: 6px; padding: 6px 12px; background: #050c12; border: 1px solid var(--line, #284052); border-radius: 8px; cursor: pointer; user-select: none; }
.chip-item.selected { background: rgba(83, 212, 255, 0.12); border-color: var(--cyan, #53d4ff); }
.font-bold { font-weight: 600; }
.mb0 { margin-bottom: 0; }
.btn-link { padding: 4px 12px; font-size: 0.85rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 6px; text-decoration: none; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }
.b-green { background: rgba(83, 229, 163, 0.15); color: #53e5a3; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
</style>
