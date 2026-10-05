<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "../api/client";

const rows = ref<any[]>([]);
const suites = ref<any[]>([]);
const selectedSuite = ref("all");

onMounted(async () => {
  rows.value = (await api("/evals/leaderboard")).items || [];
  suites.value = (await api("/evals/suites")).items || [];
});

const filteredRows = computed(() => {
  if (selectedSuite.value === "all") return rows.value;
  return rows.value.filter((r) => r.suite_id === selectedSuite.value);
});
</script>

<template>
  <div class="leaderboard-page">
    <div class="page-head">
      <div>
        <p class="crumb">评估与基准 / 排行榜</p>
        <h2>策略排行榜 (Leaderboard)</h2>
      </div>
      <div class="row">
        <label class="small muted">
          筛选评估套件：
          <select v-model="selectedSuite" class="select-sm">
            <option value="all">全部套件 (All Suites)</option>
            <option v-for="s in suites" :key="s.id" :value="s.id">{{ s.name }} ({{ s.id }})</option>
          </select>
        </label>
        <router-link to="/eval/compare" class="btn-link">📊 多策略对比</router-link>
      </div>
    </div>
    <p class="muted small">基于固定种子集测定的策略排行榜。强制显示 Wilson 95% 置信区间；严禁使用单次偶发最高分打榜。</p>

    <div class="panel" style="padding:0; overflow:hidden">
      <table>
        <thead>
          <tr>
            <th style="width:60px">排名</th>
            <th>受评模型 / 策略</th>
            <th>评估套件 (Suite)</th>
            <th>成功率与 Wilson 95% 置信区间</th>
            <th>平均回报</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, idx) in filteredRows" :key="r.id">
            <td>
              <span class="rank-badge" :class="idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : ''">
                #{{ idx + 1 }}
              </span>
            </td>
            <td>
              <div class="font-bold">
                <router-link v-if="r.model_version_id" :to="`/models/${r.model_version_id}`">{{ r.model_version_id }}</router-link>
                <span v-else>运行 checkpoint</span>
              </div>
              <div class="mono xs text-muted">report: <router-link :to="`/eval/reports/${r.id}`">{{ r.id }}</router-link></div>
            </td>
            <td>
              <span class="mono small">{{ r.suite_id }}</span>
            </td>
            <td>
              <div class="ci-wrapper">
                <div class="ci-bar-container">
                  <div class="ci-range-bar" :style="{
                    left: ((r.wilson_ci?.lower || r.metrics?.success_rate || 0) * 100) + '%',
                    width: Math.max(4, ((r.wilson_ci?.upper || r.metrics?.success_rate || 0) - (r.wilson_ci?.lower || 0)) * 100) + '%'
                  }"></div>
                  <div class="ci-point" :style="{ left: ((r.metrics?.success_rate || 0) * 100) + '%' }"></div>
                </div>
                <span class="mono small font-bold" style="min-width:140px">
                  {{ r.wilson_ci?.label || ((r.metrics?.success_rate != null ? (r.metrics.success_rate * 100).toFixed(1) + '%' : '—')) }}
                </span>
              </div>
            </td>
            <td>
              <span class="mono">{{ r.metrics?.mean_reward != null ? r.metrics.mean_reward.toFixed(2) : '—' }}</span>
            </td>
            <td>
              <router-link :to="`/eval/reports/${r.id}`" class="btn-sm">查看报告</router-link>
            </td>
          </tr>
          <tr v-if="!filteredRows.length">
            <td colspan="6" class="text-center text-muted" style="padding:2rem">暂无上榜报告数据</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.leaderboard-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.rank-badge { display: inline-flex; justify-content: center; align-items: center; width: 26px; height: 26px; border-radius: 50%; background: #16293a; font-weight: 800; font-family: var(--mono, monospace); font-size: 0.8rem; color: var(--muted, #9fb4c2); }
.rank-badge.gold { background: rgba(255, 200, 87, 0.2); color: #ffc857; border: 1px solid #ffc857; }
.rank-badge.silver { background: rgba(180, 200, 220, 0.2); color: #dbe9f2; border: 1px solid #9fb4c2; }
.rank-badge.bronze { background: rgba(255, 136, 77, 0.2); color: #ff884d; border: 1px solid #ff884d; }
.ci-wrapper { display: flex; align-items: center; gap: 12px; }
.ci-bar-container { position: relative; width: 100px; height: 8px; background: #050c12; border-radius: 4px; border: 1px solid #1d3243; }
.ci-range-bar { position: absolute; height: 100%; background: rgba(83, 229, 163, 0.35); border-radius: 2px; }
.ci-point { position: absolute; top: -2px; width: 4px; height: 12px; background: #53e5a3; border-radius: 1px; }
.font-bold { font-weight: 600; }
.text-muted { color: var(--muted, #9fb4c2); }
.select-sm { padding: 3px 8px; font-size: 0.82rem; background: #050c12; border: 1px solid #284052; border-radius: 4px; color: #edf6fb; }
.btn-sm { padding: 3px 8px; font-size: 0.8rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 4px; text-decoration: none; }
.btn-link { padding: 4px 12px; font-size: 0.85rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 6px; text-decoration: none; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }
.b-muted { background: rgba(255, 255, 255, 0.08); color: #9fb4c2; }
</style>
