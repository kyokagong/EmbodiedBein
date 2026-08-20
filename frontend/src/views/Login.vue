<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";
const user = ref("kk");
const password = ref("embodied");
const err = ref("");
const router = useRouter();
async function go() {
  try {
    await api("/auth/login", { method: "POST", body: JSON.stringify({ username: user.value, password: password.value }) });
    router.push("/");
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div class="main" style="max-width:420px;margin:10vh auto">
    <h2>登录 EmbodiedBein</h2>
    <p class="muted">单用户模式。也可用 API Key <code>eb-dev-key</code>。</p>
    <div class="row" style="flex-direction:column;align-items:stretch">
      <input v-model="user" placeholder="用户名" />
      <input v-model="password" type="password" placeholder="密码" />
      <button @click="go">进入</button>
      <p class="error">{{ err }}</p>
    </div>
  </div>
</template>
