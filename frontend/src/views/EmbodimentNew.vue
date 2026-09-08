<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const displayName = ref("Berkeley Humanoid Lite");
const rootPath = ref("/data/kk/Berkeley-Humanoid-Lite");
const err = ref("");

async function submit() {
  err.value = "";
  try {
    const row = await api("/embodiments", {
      method: "POST",
      body: JSON.stringify({ display_name: displayName.value, root_path: rootPath.value }),
    });
    router.push(`/embodiments/${row.id}`);
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <p class="crumb">本体 / 接入</p>
    <h2>登记本体目录</h2>
    <p class="muted xs">指定本地路径，平台只读扫描 URDF / gym.register / env YAML。不会写入 <code>/data/kk/*</code> 三仓。</p>
    <div class="panel" style="max-width:640px">
      <div class="form-grid">
        <span>显示名</span><input v-model="displayName" />
        <span>本地路径</span><input v-model="rootPath" />
      </div>
      <p class="muted xs" style="margin-top:.7rem">路径须落在允许前缀（默认 <code>/data/kk</code>、<code>/var/lib/eb</code>）。</p>
      <p class="error">{{ err }}</p>
      <button class="primary" style="margin-top:.8rem" @click="submit">扫描并登记</button>
    </div>
  </div>
</template>
