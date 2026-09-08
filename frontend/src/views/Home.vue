<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "../api/client";

const me = ref<any>(null);
const embodiments = ref<any[]>([]);
const trains = ref<any[]>([]);
const models = ref<any[]>([]);
const gpus = ref<any>(null);
const doctor = ref<any>(null);

onMounted(async () => {
  me.value = await api("/auth/me");
  embodiments.value = (await api("/embodiments")).items || [];
  try {
    trains.value = (await api("/train")).items || [];
  } catch {
    trains.value = [];
  }
  const m = await api("/models");
  models.value = m.items || m.versions || [];
  gpus.value = await api("/resources/gpus");
  try {
    doctor.value = await api("/envs/doctor");
  } catch {
    doctor.value = null;
  }
});

const running = computed(() => trains.value.filter((j) => ["running", "preparing", "queued"].includes(j.status)));
const gpuList = computed(() => gpus.value?.gpus || []);
const usedGpus = computed(() => gpuList.value.filter((g: any) => (g.util || g.utilization || 0) > 5).length);

function st(s: string) {
  if (s === "running" || s === "preparing") return "st-badge st-run";
  if (s === "succeeded" || s === "succeeded_pending_gate" || s === "ready") return "st-badge st-ok";
  if (s === "queued") return "st-badge st-q";
  if (s === "failed" || s === "aborted") return "st-badge st-err";
  return "st-badge";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">工作台 / 总览</p>
        <h2>工作台总览</h2>
      </div>
      <router-link class="badge" to="/train/new">＋ 发起 RLinf 训练</router-link>
    </div>
    <div class="kpis">
      <div class="card"><div class="l">运行中训练</div><div class="v">{{ running.length }}</div><div class="muted">{{ me?.principal || "—" }}</div></div>
      <div class="card"><div class="l">GPU 占用</div><div class="v">{{ usedGpus }} / {{ gpuList.length || "—" }}</div><div class="muted">{{ gpus?.hostname || "ebd doctor" }}</div></div>
      <div class="card"><div class="l">本体 / 模型</div><div class="v">{{ embodiments.length }} / {{ models.length }}</div></div>
      <div class="card">
        <div class="l">环境健康</div>
        <div class="v" :class="doctor?.healthy === false ? 'bad' : 'ok'">{{ doctor ? (doctor.healthy ? "Doctor OK" : "异常") : "—" }}</div>
      </div>
    </div>
    <div class="panel" v-if="gpuList.length">
      <h3>物理 GPU</h3>
      <div class="gpu-grid">
        <div class="gpu-box" v-for="(g, i) in gpuList" :key="i">
          <div class="row" style="justify-content:space-between">
            <b>GPU {{ g.index ?? i }}</b>
            <span class="muted xs">{{ g.util ?? g.utilization ?? 0 }}%</span>
          </div>
          <div class="gpu-bar"><i :style="{ width: Math.min(100, Number(g.util ?? g.utilization ?? 0)) + '%' }" /></div>
          <p class="muted xs">{{ g.name || g.model || "GPU" }}</p>
        </div>
      </div>
    </div>
    <div class="panel">
      <h3>最近训练</h3>
      <table v-if="trains.length">
        <thead><tr><th>名称</th><th>状态</th><th>模板</th><th>Ray</th></tr></thead>
        <tbody>
          <tr v-for="j in trains.slice(0, 8)" :key="j.id">
            <td><router-link :to="`/train/${j.id}`">{{ j.name }}</router-link><br><span class="muted xs mono">{{ j.id }}</span></td>
            <td><span :class="st(j.status)">{{ j.status }}</span></td>
            <td class="mono xs">{{ j.template_id }}</td>
            <td class="mono xs">{{ j.ray_port || "—" }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">还没有训练任务。从「训练」提交一份 RLinf 模板。</p>
    </div>
  </div>
</template>
