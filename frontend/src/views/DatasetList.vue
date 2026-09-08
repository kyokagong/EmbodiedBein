<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const rows = ref<any[]>([]);
onMounted(async () => {
  rows.value = (await api("/datasets")).items || [];
});

function gb(bytes: number | null) {
  if (bytes == null || bytes === 0) return "—";
  return (Number(bytes) / 1e9).toFixed(1) + " GB";
}

function st(s: string) {
  if (s === "ready") return "st-badge st-ok";
  if (s === "downloading" || s === "pending" || s === "extracting") return "st-badge st-run";
  if (s === "failed") return "st-badge st-err";
  return "st-badge";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">数据</p>
        <h2>数据集</h2>
      </div>
      <router-link class="badge" to="/data/new">＋ 接入数据集</router-link>
    </div>
    <table v-if="rows.length">
      <thead>
        <tr>
          <th>名称</th><th>来源</th><th>路径</th><th>规模</th><th>状态</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in rows" :key="r.id">
          <td>
            <router-link :to="`/data/${r.id}`">{{ r.name }}</router-link><br>
            <span class="muted xs">{{ r.id }}</span>
          </td>
          <td><span class="badge">{{ r.source_kind || "path" }}</span></td>
          <td class="mono xs">{{ r.dest_path || r.source_uri || r.meta?.root || "—" }}</td>
          <td>{{ gb(r.total_bytes || r.bytes) }}</td>
          <td><span :class="st(r.status || 'ready')">{{ r.status || "ready" }}</span></td>
        </tr>
      </tbody>
    </table>
    <p v-else class="muted">无数据集。接入本地路径、Hugging Face 或 HTTP。</p>
  </div>
</template>
