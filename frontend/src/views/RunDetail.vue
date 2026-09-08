<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
import LogPane from "../components/LogPane.vue";

const route = useRoute();
const data = ref<any>(null);
const metrics = ref<any>({ series: {} });
const logs = ref<string[]>([]);
const cmd = ref("");
const copiedCmd = ref(false);
const filterText = ref("");
const autoscroll = ref(true);
let ws: WebSocket | null = null;

onMounted(async () => {
  const id = route.params.id;
  try {
    data.value = await api(`/runs/${id}`);
    metrics.value = await api(`/runs/${id}/metrics`);
    logs.value = (await api(`/runs/${id}/logs?tail=200`)).lines || [];
    if (data.value?.job_id) {
      try {
        const cmdRes = await api(`/jobs/${data.value.job_id}/equivalent-command`);
        cmd.value = cmdRes.command || "";
      } catch {}
    }
  } catch (e) {
    console.error(e);
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "metrics" && msg.payload?.series) metrics.value = msg.payload;
      if (msg.type === "log" && msg.payload?.lines) logs.value = msg.payload.lines;
    } catch {}
  };
});

onUnmounted(() => ws?.close());

const filteredLogs = computed(() => {
  if (!filterText.value.trim()) return logs.value;
  return logs.value.filter((line) => line.toLowerCase().includes(filterText.value.toLowerCase()));
});

function copyCommand() {
  if (!cmd.value) return;
  navigator.clipboard?.writeText(cmd.value);
  copiedCmd.value = true;
  setTimeout(() => (copiedCmd.value = false), 2000);
}
</script>

<template>
  <div class="run-detail-page">
    <div class="page-head">
      <div>
        <p class="crumb">训练与执行 / <router-link to="/train/queue">队列</router-link> / {{ data?.id }}</p>
        <h2>
          Run {{ data?.id }}
          <span class="badge" :class="data?.status === 'running' ? 'b-cyan' : data?.status === 'finished' ? 'b-green' : 'b-amber'">
            {{ data?.status || "loading" }}
          </span>
          <span class="badge b-muted" v-if="data?.attempt">attempt #{{ data.attempt }}</span>
        </h2>
      </div>
      <div class="row">
        <button type="button" @click="copyCommand" :disabled="!cmd">
          {{ copiedCmd ? "✓ 已复制命令" : "📋 复制等价 Docker 命令" }}
        </button>
      </div>
    </div>

    <!-- 运行状态信息条 -->
    <div class="kpis">
      <div class="card">
        <div class="l">状态</div>
        <div class="v">{{ data?.status }}</div>
        <div class="muted">{{ data?.abort_reason ? '原因: ' + data.abort_reason : '正常' }}</div>
      </div>
      <div class="card">
        <div class="l">关联 Job</div>
        <div class="v"><router-link :to="`/train/queue`">{{ data?.job_id || "—" }}</router-link></div>
        <div class="muted">排队与调度记录</div>
      </div>
      <div class="card">
        <div class="l">GPU 资源</div>
        <div class="v">{{ data?.assigned_gpus?.length ? 'GPU ' + data.assigned_gpus.join(',') : 'CPU / 未占用' }}</div>
        <div class="muted">独占准入模式</div>
      </div>
      <div class="card">
        <div class="l">TSDB 曲线点</div>
        <div class="v">{{ Object.keys(metrics?.series || {}).length }} <small>标量项</small></div>
        <div class="muted">Rust collector 实时提取</div>
      </div>
    </div>

    <!-- 等价命令展开卡片 -->
    <div class="panel" v-if="cmd" style="margin-bottom:1rem">
      <div class="page-head" style="margin-bottom:0.4rem">
        <h3>等价复现命令 (Equivalent Reproducible Command)</h3>
        <button type="button" class="btn-sm" @click="copyCommand">{{ copiedCmd ? "已复制" : "复制" }}</button>
      </div>
      <pre class="cmd-box"><code>{{ cmd }}</code></pre>
    </div>

    <!-- 实时指标图表网格 -->
    <div class="panel">
      <h3>TSDB 实时标量指标</h3>
      <p class="muted small" v-if="!Object.keys(metrics?.series || {}).length">尚无指标数据流（容器可能正在初始化或加载权重）。</p>
      <div class="charts-grid" v-else>
        <div class="chart-card" v-for="(pts, tag) in metrics.series" :key="String(tag)">
          <Chart :title="String(tag)" :points="pts" />
        </div>
      </div>
    </div>

    <!-- 控制台日志面板 -->
    <div class="panel" style="margin-top:1rem">
      <div class="page-head" style="margin-bottom:0.4rem">
        <h3>容器实时控制台输出 (console.log)</h3>
        <div class="row">
          <input v-model="filterText" placeholder="过滤日志内容..." class="input-sm" />
          <label class="small muted"><input type="checkbox" v-model="autoscroll" /> 自动滚动</label>
        </div>
      </div>
      <LogPane :lines="filteredLogs" />
    </div>
  </div>
</template>

<style scoped>
.run-detail-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin-top: 0.8rem; }
.chart-card { background: #050c12; border: 1px solid #1d3243; border-radius: 8px; padding: 0.8rem; }
.cmd-box { background: #050c12; border: 1px solid #1d3243; border-radius: 8px; padding: 0.8rem; font-family: var(--mono, monospace); font-size: 0.8rem; color: #53d4ff; overflow-x: auto; white-space: pre-wrap; word-break: break-all; max-height: 140px; margin: 0; }
.input-sm { padding: 3px 8px; font-size: 0.82rem; background: #050c12; border: 1px solid #284052; border-radius: 4px; color: #edf6fb; }
.btn-sm { padding: 2px 8px; font-size: 0.78rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 4px; cursor: pointer; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.b-green { background: rgba(83, 229, 163, 0.15); color: #53e5a3; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
.b-muted { background: rgba(255, 255, 255, 0.08); color: #9fb4c2; }
</style>
