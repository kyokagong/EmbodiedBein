<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const name = ref("");
const kind = ref("path");
const uri = ref("/data/kk/berkeley_ego_data");
const dest = ref("");
const token = ref("");
const err = ref("");

async function submit() {
  err.value = "";
  try {
    const row = await api("/datasets", {
      method: "POST",
      body: JSON.stringify({
        name: name.value,
        source_kind: kind.value,
        source_uri: uri.value,
        dest_path: dest.value || null,
        hf_token: token.value || null,
      }),
    });
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
    <p class="muted xs">三源：本地路径直挂、Hugging Face、HTTP。路径必须落在白名单前缀内。</p>
    <div class="panel" style="max-width:640px">
      <div class="form-grid">
        <span>显示名</span><input v-model="name" placeholder="berkeley_ego_data" />
        <span>来源</span>
        <select v-model="kind">
          <option value="path">本地路径</option>
          <option value="huggingface">Hugging Face</option>
          <option value="http">HTTP</option>
        </select>
        <span>{{ kind === "path" ? "路径" : "URI" }}</span>
        <input v-model="uri" />
        <template v-if="kind !== 'path'">
          <span>落地目录</span>
          <input v-model="dest" placeholder="/var/lib/eb/datasets/…" />
        </template>
        <template v-if="kind === 'huggingface'">
          <span>HF token</span>
          <input v-model="token" type="password" placeholder="可选" />
        </template>
      </div>
      <p class="error">{{ err }}</p>
      <button class="primary" style="margin-top:.8rem" @click="submit">登记</button>
    </div>
  </div>
</template>
