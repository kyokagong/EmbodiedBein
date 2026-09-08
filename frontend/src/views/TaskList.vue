<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const items = ref<any[]>([]);
const draftId = ref("tsk_custom");
const envIds = ref("ee_bhl_biped_12dof_v1");
const gymId = ref("");
const err = ref("");

onMounted(async () => {
  items.value = (await api("/tasks")).items || [];
});

async function reload() {
  items.value = (await api("/tasks")).items || [];
}

async function createDraft() {
  err.value = "";
  try {
    await api("/tasks", {
      method: "POST",
      body: JSON.stringify({
        spec: {
          id: draftId.value,
          compatible_env_ids: envIds.value.split(",").map((s) => s.trim()).filter(Boolean),
          stack_bindings: gymId.value ? { isaaclab_rslrl: { gym_id: gymId.value } } : {},
          status: "draft",
        },
      }),
    });
    await reload();
  } catch (e: any) {
    err.value = e.message;
  }
}

async function publish(id: string) {
  err.value = "";
  try {
    await api(`/tasks/${id}/publish`, { method: "POST", body: "{}" });
    await reload();
  } catch (e: any) {
    err.value = e.message;
  }
}

async function remove(id: string) {
  err.value = "";
  try {
    await api(`/tasks/${id}`, { method: "DELETE" });
    await reload();
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">任务</p>
        <h2>Task <span class="badge">{{ items.length }}</span></h2>
      </div>
    </div>
    <p class="muted xs">任务是训练/评估入口，不是本体环境。不在平台编奖励或观测拼接。</p>
    <p class="error">{{ err }}</p>
    <table v-if="items.length">
      <thead>
        <tr><th>id</th><th>状态</th><th>兼容环境</th><th>Isaac gym_id</th><th></th></tr>
      </thead>
      <tbody>
        <tr v-for="t in items" :key="t.id">
          <td class="mono xs">{{ t.id }}</td>
          <td><span class="badge">{{ t.status }}</span></td>
          <td class="mono xs">{{ (t.compatible_env_ids || []).join(", ") }}</td>
          <td class="mono xs">{{ t.stack_bindings?.isaaclab_rslrl?.gym_id || t.stack_bindings?.rlinf?.config_name || "—" }}</td>
          <td>
            <button v-if="t.status === 'draft'" type="button" @click="publish(t.id)">发布</button>
            <button v-if="t.status === 'draft'" type="button" @click="remove(t.id)">删除</button>
          </td>
        </tr>
      </tbody>
    </table>
    <div class="panel" style="margin-top:1rem;max-width:640px">
      <h3>新建草稿</h3>
      <div class="form-grid">
        <span>id</span><input v-model="draftId" />
        <span>兼容 ee_*</span><input v-model="envIds" />
        <span>gym_id</span><input v-model="gymId" placeholder="可选" />
      </div>
      <button type="button" style="margin-top:.6rem" @click="createDraft">创建草稿</button>
    </div>
  </div>
</template>
