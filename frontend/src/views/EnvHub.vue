<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const stack = ref<any>(null);
const images = ref<any[]>([]);
const doctor = ref<any>(null);
const err = ref("");

onMounted(load);

async function load() {
  stack.value = await api("/envs/stack");
  images.value = (await api("/envs/images")).items || [];
}

async function runDoctor() {
  err.value = "";
  try {
    doctor.value = await api("/envs/doctor");
  } catch (e: any) {
    err.value = e.message;
  }
}

async function setDefault(id: string) {
  await api(`/envs/images/${id}/default`, { method: "POST", body: "{}" });
  await load();
}

function gb(n: number | null) {
  if (!n) return "—";
  return (n / 1e9).toFixed(1) + " GB";
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">环境 / Isaac Lab</p>
        <h2>环境与镜像</h2>
      </div>
      <button class="primary" type="button" @click="runDoctor">运行 Doctor</button>
    </div>
    <p class="error">{{ err }}</p>
    <div class="panel">
      <h3>{{ stack?.display_name || "Isaac Lab" }}</h3>
      <p class="muted xs">平台统一用 Isaac Lab 作为仿真运行时。训练容器只读挂载代码仓，可写区仅 <code>/var/lib/eb/runs/&lt;id&gt;/</code>。</p>
      <dl class="dl">
        <dt>默认镜像</dt><dd>{{ stack?.default_image }}</dd>
        <dt>Nucleus</dt><dd>{{ stack?.cache?.nucleus }}</dd>
        <dt>ov_data</dt><dd>{{ stack?.cache?.ov_data }}</dd>
      </dl>
    </div>
    <div class="panel" v-if="doctor">
      <h3>Doctor {{ doctor.healthy ? "通过" : "未全部通过" }}</h3>
      <div class="grid2">
        <div class="card" v-for="c in doctor.checks" :key="c.id">
          <div class="l">{{ c.label }}</div>
          <div class="v" :class="c.ok ? 'ok' : 'bad'">{{ c.ok ? "OK" : "失败" }}</div>
          <p class="muted xs" v-if="typeof c.detail === 'string'">{{ c.detail }}</p>
        </div>
      </div>
    </div>
    <div class="panel">
      <h3>镜像目录</h3>
      <p class="muted xs">ebd 当前没有 Docker pull API，状态来自目录登记。提交训练时绑定镜像名。</p>
      <table>
        <thead><tr><th>镜像</th><th>角色</th><th>Digest</th><th>体积</th><th>状态</th><th></th></tr></thead>
        <tbody>
          <tr v-for="im in images" :key="im.id">
            <td class="mono xs">{{ im.name }}</td>
            <td>{{ im.role }}</td>
            <td class="mono xs">{{ im.digest_short || "—" }}</td>
            <td>{{ gb(im.size_bytes) }}</td>
            <td><span class="st-badge" :class="im.is_default ? 'st-ok' : ''">{{ im.status }}</span></td>
            <td>
              <span v-if="im.is_default" class="muted xs">默认</span>
              <button v-else type="button" @click="setDefault(im.id)">设为默认</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
