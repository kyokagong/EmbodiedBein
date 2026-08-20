<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
import EmptyState from "../components/EmptyState.vue";
const rows = ref<any[]>([]);
const err = ref("");
async function load() {
  rows.value = (await api("/pipelines")).items || [];
}
async function start(id: string) {
  err.value = "";
  try {
    await api(`/pipelines/${id}/start`, { method: "POST" });
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}
onMounted(load);
</script>
<template>
  <div>
    <h2>流水线</h2>
    <p class="muted">steps[] 顺序提交，下一步等 SUCCEEDED。点开始后由调度器推进。</p>
    <p class="error">{{ err }}</p>
    <EmptyState v-if="!rows.length" title="无流水线">bootstrap 会写入 scan→slice 示例。</EmptyState>
    <EbTable
      v-else
      :columns="[{key:'id',label:'id'},{key:'name',label:'name'},{key:'status',label:'status'}]"
      :rows="rows"
    />
    <div class="row" v-for="r in rows" :key="r.id">
      <button @click="start(r.id)">开始 {{ r.id }}</button>
      <span class="muted">jobs: {{ (r.job_ids || []).join(', ') || '—' }}</span>
    </div>
  </div>
</template>
