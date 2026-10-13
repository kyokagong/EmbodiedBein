<script setup lang="ts">
import { ref } from "vue";
import { api } from "../api/client";

const props = defineProps<{ path: string; name: string; disabled?: boolean }>();
const emit = defineEmits<{ deleted: [] }>();
const busy = ref(false);
const error = ref("");
async function remove() {
  if (busy.value || props.disabled) return;
  busy.value = true;
  error.value = "";
  try {
    const plan = await api(`${props.path}/deletion-preview`);
    if (plan.blockers?.length) {
      error.value = `仍被引用，需先处理：${plan.blockers.slice(0, 8).map((b: any) => b.name || b.id).join("、")}${plan.blockers.length > 8 ? ` 等 ${plan.blockers.length} 项` : ""}`;
      return;
    }
    const files = plan.paths?.length ? `\n同时永久删除文件：\n${plan.paths.join("\n")}` : "\n删除登记记录。";
    const kept = plan.preserved_paths?.length ? `\n保留外部或共享源文件：\n${plan.preserved_paths.join("\n")}` : "";
    if (!window.confirm(`永久删除「${props.name}」？${files}${kept}\n审计与血缘记录保留，删除文件不可恢复。`)) return;
    await api(props.path, { method: "DELETE" });
    emit("deleted");
  } catch (e: any) {
    try {
      const detail = JSON.parse(e.message).detail;
      error.value = typeof detail === "string" ? detail : detail?.message || e.message;
    } catch { error.value = e.message; }
  } finally { busy.value = false; }
}
</script>
<template>
  <span>
    <button type="button" class="delete-button" :disabled="busy || disabled" :title="disabled ? '请先取消活动任务，结束后再删除' : '预览并删除'" @click="remove">{{ busy ? "处理中…" : "删除" }}</button>
    <span v-if="error" class="error xs" role="alert">{{ error }}</span>
  </span>
</template>
<style scoped>
.delete-button { color: #b42318; }
</style>
