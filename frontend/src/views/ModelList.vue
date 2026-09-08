<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const items = ref<any[]>([]);
const embodiments = ref<any[]>([]);
const show = ref(false);
const name = ref("");
const path = ref("");
const embodimentId = ref("");
const err = ref("");

onMounted(load);

async function load() {
  const m = await api("/models");
  items.value = m.items || [];
  embodiments.value = (await api("/embodiments")).items || [];
}

function gb(n: number) {
  if (!n) return "—";
  return n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB";
}

async function register() {
  err.value = "";
  try {
    await api("/weights", {
      method: "POST",
      body: JSON.stringify({ name: name.value, path: path.value, embodiment_id: embodimentId.value || null }),
    });
    show.value = false;
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">模型</p>
        <h2>模型与血缘</h2>
      </div>
      <button class="primary" type="button" @click="show = !show">＋ 登记基座</button>
    </div>
    <div class="panel" v-if="show">
      <h3>登记基座权重</h3>
      <div class="form-grid">
        <span>名称</span><input v-model="name" />
        <span>路径</span><input v-model="path" placeholder="/data/kk/… 或 /var/lib/eb/…" />
        <span>本体</span>
        <select v-model="embodimentId">
          <option value="">（可选）</option>
          <option v-for="e in embodiments" :key="e.id" :value="e.id">{{ e.display_name }}</option>
        </select>
      </div>
      <p class="error">{{ err }}</p>
      <button class="primary" style="margin-top:.6rem" @click="register">保存</button>
    </div>
    <table v-if="items.length">
      <thead>
        <tr><th>名称</th><th>类型</th><th>路径</th><th>体积</th><th>训练</th></tr>
      </thead>
      <tbody>
        <tr v-for="w in items" :key="w.id">
          <td>
            <router-link :to="`/models/${w.id}`">{{ w.name }}</router-link>
            <br><span class="muted xs mono">{{ w.id }}</span>
          </td>
          <td><span class="st-badge" :class="w.kind === 'base' ? 'st-ok' : 'st-run'">{{ w.kind }}</span></td>
          <td class="mono xs">{{ w.path }}</td>
          <td class="xs">{{ gb(w.size_bytes) }}</td>
          <td class="mono xs">{{ w.train_job_id || "—" }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="muted">还没有权重。登记基座，或从训练详情勾选 checkpoint。</p>
  </div>
</template>
