<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const name = ref("");
const kind = ref("path");
const uri = ref("/data/kk/berkeley_ego_data");
const dest = ref("");
const revision = ref("");
const environment = ref("ee_libero_franka_delta_v1");
const usage = ref("train");
const err = ref("");
const liberoRevision = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f";
const robotwin = ref<any>({ tasks: [], embodiments: [], download_profiles: [] });
const robotwinMode = ref("profile");
const profileId = ref("aloha_lerobot_v21");
const robotwinTask = ref("adjust_bottle");
const robotwinScenario = ref("clean");
const robotwinEmbodiment = ref("aloha-agilex");
const selectedRobotwinProfile = computed(() => robotwin.value.download_profiles.find((p: any) => p.id === profileId.value));

onMounted(async () => {
  try { robotwin.value = await api("/dataset-sources/robotwin"); }
  catch (e: any) { err.value = e.message; }
});

async function submit() {
  err.value = "";
  try {
    if (kind.value === "robotwin") {
      const row = await api("/datasets/robotwin", { method: "POST", body: JSON.stringify({
        name: name.value,
        ...(robotwinMode.value === "profile" ? { profile_id: profileId.value } : {
          task: robotwinTask.value, scenario: robotwinScenario.value, embodiment: robotwinEmbodiment.value,
        }),
      }) });
      if (row.status === "registered" || row.status === "failed" || row.status === "invalid")
        await api(`/datasets/${row.id}/validate`, { method: "POST", body: "{}" });
      router.push(`/data/${row.id}`);
      return;
    }
    if (kind.value === "rpo_amp_official") {
      const row = await api("/datasets/rpo-amp-official", { method: "POST",
        body: JSON.stringify({ name: name.value || "RoboParty RPO AMP stand/walk probe" }) });
      router.push(`/data/${row.id}`);
      return;
    }
    const benchmark = kind.value === "libero_benchmark";
    const row = await api("/datasets", {
      method: "POST",
      body: JSON.stringify({
        name: name.value,
        source_kind: kind.value,
        source_uri: benchmark ? "RLinf/LIBERO-assets" : uri.value,
        dest_path: benchmark ? null : dest.value || null,
        revision: benchmark ? liberoRevision : revision.value || null,
        embodiment_env_id: benchmark ? "ee_libero_franka_delta_v1" : environment.value,
        usage: benchmark ? "eval" : usage.value,
        purpose: environment.value === "ee_rpo_v2_amp_locomotion_23dof_v1" ? "motion_prior" : null,
      }),
    });
    if (benchmark && row.status !== "validated") await api(`/datasets/${row.id}/validate`, { method: "POST", body: "{}" });
    router.push(`/data/${row.id}`);
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <p class="crumb">数据 / 接入</p>
    <h2>接入数据集</h2>
    <p class="muted xs">LIBERO 测试场景含任务定义、初始状态与仿真资产；固定版本下载后用于闭环评估。</p>
    <div class="panel" style="max-width:640px">
      <div class="form-grid">
        <span>显示名</span><input v-model="name" placeholder="berkeley_ego_data" />
        <span v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)">本体环境 ID</span><input v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)" v-model="environment" />
        <span v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)">用途</span><select v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)" v-model="usage"><option value="train">训练集</option><option value="eval">评估集</option></select>
        <span v-if="environment === 'ee_rpo_v2_amp_locomotion_23dof_v1'">消费用途</span><span v-if="environment === 'ee_rpo_v2_amp_locomotion_23dof_v1'">AMP 动作先验（需本地训练路径与验证）</span>
        <span>来源</span>
        <select v-model="kind">
          <option value="path">本地路径</option>
          <option value="huggingface">Hugging Face</option>
          <option value="robotwin">RoboTwin 2.0（固定版本）</option>
          <option value="rpo_amp_official">RoboParty RPO 官方动作（固定版本）</option>
          <option value="libero_benchmark">LIBERO 测试场景（固定版本）</option>

        </select>
        <template v-if="kind === 'robotwin'">
          <span>数据类型</span><select v-model="robotwinMode"><option value="profile">训练转换包</option><option value="raw">原生任务压缩包</option></select>
          <template v-if="robotwinMode === 'profile'">
            <span>数据包</span><select v-model="profileId"><option v-for="p in robotwin.download_profiles" :key="p.id" :value="p.id">{{ p.label }}</option></select>
            <span>原生本体</span><span>{{ selectedRobotwinProfile?.native_embodiment || '等待目录' }}</span>
          </template>
          <template v-else>
            <span>任务</span><select v-model="robotwinTask"><option v-for="t in robotwin.tasks" :key="t" :value="t">{{ t }}</option></select>
            <span>场景</span><select v-model="robotwinScenario"><option value="clean">clean · 50</option><option value="randomized">randomized · 500</option></select>
            <span>原生本体</span><select v-model="robotwinEmbodiment"><option v-for="e in robotwin.embodiments" :key="e.id" :value="e.id" :disabled="e.download_ready === false">{{ e.label }}{{ e.download_ready === false ? ' · 待接口探测' : '' }}</option></select>
          </template>
          <span>固定来源</span><span class="mono xs">{{ robotwin.repo_id }} @ {{ robotwin.revision }}</span>
        </template>
        <span v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)">{{ kind === "path" ? "路径" : "URI" }}</span>
        <input v-if="!['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)" v-model="uri" />
        <template v-if="kind !== 'path' && !['libero_benchmark', 'robotwin', 'rpo_amp_official'].includes(kind)">
          <span>落地目录</span>
          <input v-model="dest" placeholder="/var/lib/eb/datasets/…" />
        </template>
        <template v-if="kind === 'huggingface'">
          <span>Revision</span>
          <input v-model="revision" placeholder="40 位 commit SHA" />
        </template>
        <template v-if="kind === 'libero_benchmark'"><span>来源版本</span><span class="mono xs">RLinf/LIBERO-assets @ {{ liberoRevision }}</span></template>
      </div>
      <p class="error">{{ err }}</p>
      <button class="primary" style="margin-top:.8rem" @click="submit">{{ kind === 'rpo_amp_official' ? '登记并转换' : ['libero_benchmark', 'robotwin'].includes(kind) ? '登记并下载' : '登记' }}</button>
    </div>
  </div>
</template>
