<script setup lang="ts">
import { ref } from "vue";
import { api } from "../api/client";
const src = ref("/data/kk/berkeley_ego_data");
const out = ref<any>(null);
const err = ref("");
async function convert() {
  try {
    out.value = await api("/jobs", { method: "POST", body: JSON.stringify({ spec: { kind: "data.convert", contract_id: "ec_bhl_fixedhand_10dof_v1", params: { src: src.value } } }) });
  } catch (e: any) { err.value = e.message; }
}
async function retarget() {
  try {
    out.value = await api("/jobs", { method: "POST", body: JSON.stringify({ spec: { kind: "data.retarget", contract_id: "ec_bhl_fixedhand_10dof_v1", params: { src: src.value } } }) });
  } catch (e: any) { err.value = e.message; }
}
</script>
<template>
  <div>
    <h2>转换 / 重定向</h2>
    <p class="muted">data.convert / data.retarget 走 tools 镜像。IK 不可行会 failed，不会假成功。</p>
    <input v-model="src" style="width:100%" />
    <div class="row">
      <button @click="convert">convert</button>
      <button @click="retarget">retarget</button>
    </div>
    <p class="error">{{ err }}</p>
    <pre v-if="out">{{ JSON.stringify(out, null, 2) }}</pre>
  </div>
</template>
