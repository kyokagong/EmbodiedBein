<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const wb = ref<any>(null);
const prefixes = ref("");
const rlinf = ref("");
const image = ref("");
const threshold = ref(10);
const msg = ref("");
const err = ref("");

onMounted(async () => {
  try {
    wb.value = await api("/workbench/settings");
    prefixes.value = (wb.value.allowed_path_prefixes || []).join(", ");
    rlinf.value = wb.value.rlinf_repo_path || "";
    image.value = wb.value.default_train_image || "";
    threshold.value = wb.value.disk_alert_threshold_gb || 10;
  } catch (e: any) {
    err.value = e.message;
  }
});

async function save() {
  err.value = "";
  try {
    wb.value = await api("/workbench/settings", {
      method: "PATCH",
      body: JSON.stringify({
        allowed_path_prefixes: prefixes.value.split(",").map((s) => s.trim()).filter(Boolean),
        rlinf_repo_path: rlinf.value,
        default_train_image: image.value,
        disk_alert_threshold_gb: Number(threshold.value),
      }),
    });
    msg.value = "已保存";
    setTimeout(() => (msg.value = ""), 2000);
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">设置</p>
        <h2>系统设置</h2>
      </div>
    </div>
    <p class="muted xs">路径白名单、RLinf 仓与默认训练镜像。不会写 <code>/data/kk/*</code> 三仓。</p>
    <p class="ok" v-if="msg">{{ msg }}</p>
    <p class="error">{{ err }}</p>
    <div class="panel" style="max-width:720px">
      <div class="form-grid">
        <span>允许前缀</span>
        <input v-model="prefixes" placeholder="/data/kk, /var/lib/eb" />
        <span>RLinf 路径</span>
        <input v-model="rlinf" />
        <span>默认镜像</span>
        <input v-model="image" />
        <span>磁盘告警 GB</span>
        <input type="number" v-model.number="threshold" />
      </div>
      <button class="primary" style="margin-top:1rem" @click="save">保存</button>
    </div>
  </div>
</template>
