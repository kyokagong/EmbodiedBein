<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const items = ref<any[]>([]);
const err = ref("");

onMounted(async () => {
  try {
    items.value = (await api("/train")).items || [];
  } catch (e: any) {
    err.value = e.message;
  }
});

function st(s: string) {
  if (s === "running" || s === "preparing") return "st-badge st-run";
  if (s === "succeeded" || s === "ready") return "st-badge st-ok";
  if (s === "queued") return "st-badge st-q";
  if (s === "failed" || s === "aborted") return "st-badge st-err";
  return "st-badge";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">训练</p>
        <h2>训练任务</h2>
      </div>
      <router-link class="badge" to="/train/new">＋ 新建 RLinf 训练</router-link>
    </div>
    <p class="error">{{ err }}</p>
    <table v-if="items.length">
      <thead>
        <tr>
          <th>名称</th><th>状态</th><th>模板</th><th>本体</th><th>GPU</th><th>Ray</th><th>创建</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="j in items" :key="j.id">
          <td>
            <router-link :to="`/train/${j.id}`">{{ j.name }}</router-link>
            <br><span class="muted xs mono">{{ j.id }}</span>
          </td>
          <td><span :class="st(j.status)">{{ j.status }}</span></td>
          <td class="mono xs">{{ j.template_id }}</td>
          <td class="mono xs">{{ j.embodiment_id }}</td>
          <td class="xs">{{ (j.gpus_allocated || []).join(", ") || "—" }}</td>
          <td class="mono xs">{{ j.ray_port || "—" }}</td>
          <td class="xs muted">{{ (j.created_at || "").replace("T", " ").slice(0, 16) }}</td>
        </tr>
      </tbody>
    </table>
    <div v-else class="empty-box">
      <b>还没有训练任务</b>
      本版只提交 RLinf 三模板：WBC PPO / Reach SFT / Collect。
    </div>
  </div>
</template>
