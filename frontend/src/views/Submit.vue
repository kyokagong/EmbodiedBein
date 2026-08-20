<script setup lang="ts">
import { ref } from "vue";
import { api } from "../api/client";
const spec = ref({
  kind: "train.rl",
  contract_id: "ec_bhl_biped_12dof_v1",
  name: "biped-vel",
  params: { task: "Velocity-Berkeley-Humanoid-Lite-Biped-v0", num_envs: 4096, max_iterations: 6000, seed: 0 },
  resources: { gpus: 1 },
});
const estimate = ref<any>(null);
const result = ref<any>(null);
const err = ref("");
async function est() {
  estimate.value = await api("/capacity/estimate", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
}
async function submit() {
  err.value = "";
  try {
    result.value = await api("/jobs", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <h2>提交任务</h2>
    <p class="muted">G1 契约校验 + 容量估算器。提交后进入队列，由调度器准入。</p>
    <pre>{{ JSON.stringify(spec, null, 2) }}</pre>
    <div class="row">
      <button @click="est">估算显存</button>
      <button @click="submit">提交</button>
    </div>
    <p class="error">{{ err }}</p>
    <pre v-if="estimate">{{ JSON.stringify(estimate, null, 2) }}</pre>
    <pre v-if="result">{{ JSON.stringify(result, null, 2) }}</pre>
  </div>
</template>
