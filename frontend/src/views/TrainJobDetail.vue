<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";

const route = useRoute();
const job = ref<any>(null);
const ckpts = ref<any[]>([]);
const metrics = ref<any>({ series: {} });
const logs = ref<string[]>([]);
const err = ref("");
const runId = ref("");
let timer: number | null = null;
let ws: WebSocket | null = null;

onMounted(load);
onUnmounted(() => {
  if (timer) window.clearInterval(timer);
  ws?.close();
});

async function load() {
  const id = route.params.id as string;
  job.value = await api(`/train/${id}`);
  ckpts.value = (await api(`/train/${id}/checkpoints`)).items || [];
  if (job.value.job_id) {
    try {
      const j = await api(`/jobs/${job.value.job_id}`);
      const run = (j.runs || [])[0];
      if (run?.id) {
        runId.value = run.id;
        try {
          metrics.value = await api(`/runs/${run.id}/metrics`);
          logs.value = (await api(`/runs/${run.id}/logs?tail=200`)).lines || [];
        } catch {}
        attachWs(run.id);
      }
    } catch {}
  }
  if (!timer) timer = window.setInterval(load, 8000);
}

function attachWs(id: string) {
  if (ws) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "metrics" && msg.payload?.series) metrics.value = msg.payload;
      if (msg.type === "log" && msg.payload?.lines) logs.value = msg.payload.lines;
    } catch {}
  };
}

function series(name: string) {
  const s = metrics.value?.series?.[name] || metrics.value?.series?.reward || [];
  return s;
}

async function register(p: any) {
  err.value = "";
  try {
    const row = await api(`/train/${job.value.id}/register-model`, {
      method: "POST",
      body: JSON.stringify({ path: p.path, name: p.name }),
    });
    err.value = `已登记 ${row.id}`;
  } catch (e: any) {
    err.value = e.message;
  }
}

async function cancel() {
  if (!job.value?.job_id) return;
  await api(`/jobs/${job.value.job_id}/cancel`, { method: "POST", body: "{}" });
  await load();
}

function st(s: string) {
  if (s === "running" || s === "preparing") return "st-badge st-run";
  if (s === "succeeded") return "st-badge st-ok";
  if (s === "queued") return "st-badge st-q";
  if (s === "failed" || s === "aborted") return "st-badge st-err";
  return "st-badge";
}

function gb(n: number) {
  if (!n) return "—";
  return n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB";
}
</script>
<template>
  <div v-if="job">
    <div class="page-head">
      <div>
        <p class="crumb">训练 / {{ job.id }}</p>
        <h2>{{ job.name }} <span :class="st(job.status)">{{ job.status }}</span></h2>
      </div>
      <div class="row">
        <button v-if="job.job_id && ['queued','running','preparing'].includes(job.status)" type="button" @click="cancel">终止</button>
        <router-link v-if="runId" class="badge" :to="`/train/runs/${runId}`">Run 详情</router-link>
      </div>
    </div>
    <p class="error">{{ err }}</p>
    <p class="muted xs mono">
      镜像 {{ job.image_digest }} · Ray {{ job.ray_port }} · 输出 {{ job.run_dir }}
    </p>
    <div class="grid2">
      <div class="panel">
        <h3>指标</h3>
        <Chart :points="series('reward')" title="reward" />
        <p class="muted xs" v-if="!series('reward').length">尚无 tfevents。任务跑起来后这里画曲线。</p>
      </div>
      <div class="panel">
        <h3>Checkpoint</h3>
        <div v-for="c in ckpts" :key="c.path" class="row" style="justify-content:space-between;margin:.35rem 0">
          <div>
            <b class="mono">{{ c.name }}</b>
            <div class="muted xs">{{ gb(c.bytes) }}</div>
          </div>
          <button type="button" @click="register(c)">登记为模型</button>
        </div>
        <p v-if="!ckpts.length" class="muted xs">还没有 .pt / .pth。成功后可在此勾选登记。</p>
      </div>
    </div>
    <div class="panel">
      <h3>Hydra Overrides</h3>
      <pre class="preview-box">{{ (job.hydra_overrides || []).join("\n") }}</pre>
    </div>
    <div class="panel">
      <h3>日志</h3>
      <pre class="preview-box" style="max-height:280px">{{ logs.join("\n") || "等待容器日志…" }}</pre>
    </div>
  </div>
</template>
