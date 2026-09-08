<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";

const route = useRoute();
const data = ref<any>(null);
const tree = ref<any>(null);
const dir = ref("");
const media = ref("");
const err = ref("");

onMounted(load);

async function load() {
  data.value = await api(`/datasets/${route.params.id}`);
  await loadTree("");
}

async function loadTree(path: string) {
  dir.value = path;
  try {
    tree.value = await api(`/datasets/${route.params.id}/tree?path=${encodeURIComponent(path)}&page=1&page_size=80`);
    err.value = "";
  } catch (e: any) {
    tree.value = null;
    err.value = e.message;
  }
}

function open(item: any) {
  if (item.is_dir) {
    loadTree(item.path);
    return;
  }
  const lower = item.name.toLowerCase();
  if (lower.endsWith(".mp4") || lower.endsWith(".webm") || lower.endsWith(".mkv")) {
    media.value = `/api/v1/datasets/${route.params.id}/media?path=${encodeURIComponent(item.path)}`;
  }
}

function up() {
  if (!dir.value) return;
  const parts = dir.value.split("/").filter(Boolean);
  parts.pop();
  loadTree(parts.join("/"));
}

function gb(n: number | null) {
  if (!n) return "—";
  return (n / 1e9).toFixed(2) + " GB";
}
</script>
<template>
  <div v-if="data">
    <div class="page-head">
      <div>
        <p class="crumb">数据 / {{ data.id }}</p>
        <h2>{{ data.name }} <span class="badge">{{ data.status || "ready" }}</span></h2>
      </div>
      <router-link class="badge" to="/data">← 列表</router-link>
    </div>
    <dl class="dl">
      <dt>来源</dt><dd>{{ data.source_kind || "path" }} · {{ data.source_uri || "—" }}</dd>
      <dt>落地</dt><dd>{{ data.dest_path || data.meta?.root || "—" }}</dd>
      <dt>体积</dt><dd>{{ gb(data.total_bytes || data.overview?.bytes) }}</dd>
    </dl>
    <p class="error xs">{{ err }}</p>
    <div class="grid2">
      <div class="panel">
        <h3>惰性文件树</h3>
        <div class="row" style="margin-bottom:.4rem">
          <button type="button" :disabled="!dir" @click="up">上级</button>
          <span class="mono xs muted">{{ dir || "/" }}</span>
        </div>
        <div v-if="tree">
          <div class="tree-row" v-for="it in tree.items" :key="it.path" @click="open(it)">
            <span>{{ it.is_dir ? "📁" : "📄" }}</span>
            <span>{{ it.name }}</span>
            <span class="muted xs" v-if="it.bytes">{{ it.bytes }} B</span>
          </div>
          <p class="muted xs" v-if="!tree.items?.length">空目录</p>
        </div>
        <p v-else class="muted xs">无法列出目录。确认 dest_path 在白名单内且可读。</p>
      </div>
      <div class="panel">
        <h3>媒体预览</h3>
        <video v-if="media" :src="media" controls style="width:100%;border-radius:8px;background:#000" />
        <p v-else class="muted xs">点开 mp4 / webm 在此播放（走 HTTP Range）。</p>
      </div>
    </div>
  </div>
</template>
