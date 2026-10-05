<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
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
const err = ref("");
let copyTimer: ReturnType<typeof setTimeout> | undefined;

watch(() => String(route.params.id || ""), (id, _previous, onCleanup) => {
  const controller = new AbortController();
  const signal = controller.signal;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let ws: WebSocket | null = null;
  data.value = null;
  metrics.value = { series: {} };
  logs.value = [];
  cmd.value = "";
  copiedCmd.value = false;
  err.value = "";
  clearTimeout(copyTimer);
  onCleanup(() => {
    controller.abort();
    clearTimeout(timer);
    clearTimeout(copyTimer);
    ws?.close();
  });
  if (!id) return;

  async function refreshState() {
    try {
      const current = await api(`/runs/${id}`, { signal });
      if (signal.aborted) return;
      data.value = current;
      err.value = "";
    } catch (e: any) {
      if (!signal.aborted) err.value = e.message;
    } finally {
      if (!signal.aborted && !["finished", "aborted"].includes(data.value?.status)) {
        timer = setTimeout(refreshState, 5000);
      }
    }
  }

  async function connectStreams() {
    const initial = await Promise.allSettled([
      api(`/runs/${id}/metrics`, { signal }),
      api(`/runs/${id}/logs?tail=200`, { signal }),
    ]);
    if (signal.aborted) return;
    if (initial[0].status === "fulfilled") metrics.value = initial[0].value;
    if (initial[1].status === "fulfilled") logs.value = initial[1].value.lines || [];
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}`);
    ws.onmessage = (ev) => {
      if (signal.aborted) return;
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "metrics" && msg.payload?.series) metrics.value = msg.payload;
        if (msg.type === "log" && msg.payload?.lines) logs.value = msg.payload.lines;
      } catch {}
    };
  }

  async function loadCommand() {
    await refreshState();
    if (signal.aborted || !data.value?.job_id) return;
    try {
      const result = await api(`/jobs/${data.value.job_id}/equivalent-command`, { signal });
      if (!signal.aborted) cmd.value = result.command || "";
    } catch {}
  }
  void loadCommand();
  void connectStreams();
}, { immediate: true });

onUnmounted(() => clearTimeout(copyTimer));

const filteredLogs = computed(() => {
  if (!filterText.value.trim()) return logs.value;
  return logs.value.filter((line) => line.toLowerCase().includes(filterText.value.toLowerCase()));
});

function copyCommand() {
  if (!cmd.value) return;
  navigator.clipboard?.writeText(cmd.value);
  copiedCmd.value = true;
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => (copiedCmd.value = false), 2000);
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
          {{ copiedCmd ? "✓ 已复制命令" : "📋 复制当前配置命令" }}
        </button>
      </div>
    </div>

    <p class="error" role="alert">{{ err }}</p>
    <p class="muted small">运行期间每 5 秒刷新状态，日志与指标通过实时连接更新。</p>
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
        <div class="v">{{ data?.gpu_ids?.length ? 'GPU ' + data.gpu_ids.join(',') : 'CPU / 未占用' }}</div>
        <div class="muted">本次 Run 的 GPU 分配记录</div>
      </div>
      <div class="card">
        <div class="l">TSDB 曲线点</div>
        <div class="v">{{ Object.keys(metrics?.series || {}).length }} <small>标量项</small></div>
        <div class="muted">Rust collector 实时提取</div>
      </div>
    </div>

    <div class="panel" v-if="data?.metrics_summary?.eval_report_id || data?.metrics_summary?.checkpoint" style="margin-bottom:1rem">
      <h3>本次运行产物</h3>
      <router-link v-if="data.metrics_summary.eval_report_id" :to="`/eval/reports/${data.metrics_summary.eval_report_id}`">查看本次评估报告</router-link>
      <p v-if="data.metrics_summary.qualification" class="muted small">验收状态：{{ data.metrics_summary.qualification }}</p>
      <div v-if="data.metrics_summary.checkpoint">
        <p>完整模型目录 · step {{ data.metrics_summary.checkpoint.step }}</p>
        <code style="overflow-wrap:anywhere">{{ data.metrics_summary.checkpoint.path }}</code>
      </div>
    </div>

    <!-- 等价命令展开卡片 -->
    <div class="panel" v-if="cmd" style="margin-bottom:1rem">
      <div class="page-head" style="margin-bottom:0.4rem">
        <h3>当前配置命令</h3>
        <button type="button" class="btn-sm" @click="copyCommand">{{ copiedCmd ? "已复制" : "复制" }}</button>
      </div>
      <p class="muted small">按当前适配器生成；历史运行以其镜像指纹和 resolved-config.yaml 为准。</p>
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
