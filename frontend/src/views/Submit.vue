<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";
import GatePanel from "../components/GatePanel.vue";

const contracts = [
  { id: "ec_bhl_biped_12dof_v1", task: "Velocity-Berkeley-Humanoid-Lite-Biped-v0" },
  { id: "ec_bhl_humanoid_22dof_v1", task: "Velocity-Berkeley-Humanoid-Lite-v0" },
  { id: "ec_bhl_wbc_22dof_v1", task: "Velocity-Berkeley-Humanoid-Lite-WBC-v0" },
];
const spec = ref({
  kind: "train.rl",
  contract_id: "ec_bhl_biped_12dof_v1",
  name: "biped-smoke",
  params: { task: "Velocity-Berkeley-Humanoid-Lite-Biped-v0", num_envs: 64, max_iterations: 5, seed: 0 },
  resources: { gpus: 1, shm_gb: 2 },
});
const estimate = ref<any>(null);
const gate = ref<any>(null);
const eq = ref("");
const result = ref<any>(null);
const err = ref("");
const router = useRouter();

function onContract() {
  const c = contracts.find((x) => x.id === spec.value.contract_id);
  if (c) spec.value.params.task = c.task;
}

async function est() {
  estimate.value = await api("/capacity/estimate", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
  gate.value = await api(`/contracts/${spec.value.contract_id}/validate`, {
    method: "POST",
    body: JSON.stringify({ spec: spec.value }),
  });
}
async function submit() {
  err.value = "";
  try {
    result.value = await api("/jobs", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
    if (result.value?.id) router.push("/train/queue");
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <h2>提交任务</h2>
    <p class="muted">冒烟默认 64 env × 5 iter；生产可改回 4096 / 6000。G1 与容量估算必须先看。</p>
    <div class="row" style="flex-direction:column;align-items:stretch;max-width:640px">
      <label>契约
        <select v-model="spec.contract_id" @change="onContract">
          <option v-for="c in contracts" :key="c.id" :value="c.id">{{ c.id }}</option>
        </select>
      </label>
      <label>task <input v-model="spec.params.task" /></label>
      <label>num_envs <input type="number" v-model.number="spec.params.num_envs" /></label>
      <label>max_iterations <input type="number" v-model.number="spec.params.max_iterations" /></label>
      <label>seed <input type="number" v-model.number="spec.params.seed" /></label>
      <label>GPUs <input type="number" v-model.number="spec.resources.gpus" /></label>
    </div>
    <div class="row">
      <button @click="est">G1 预检 / 估算</button>
      <button @click="submit">提交</button>
    </div>
    <p class="error">{{ err }}</p>
    <GatePanel :result="gate" />
    <div class="card" v-if="estimate">
      <div class="l">容量</div>
      <div class="v">{{ estimate.verdict }}</div>
      <p class="muted">peak {{ estimate.per_gpu_peak_gb }} GB</p>
    </div>
    <pre v-if="result">{{ JSON.stringify(result, null, 2) }}</pre>
  </div>
</template>
