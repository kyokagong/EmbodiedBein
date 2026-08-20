<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
const route = useRoute();
const ep = ref<any>(null);
const video = ref<HTMLVideoElement | null>(null);
const t = ref(0);
onMounted(async () => {
  const ds = await api(`/datasets/${route.params.id}`);
  const vid = ds.versions?.[0]?.id;
  if (vid) ep.value = await api(`/datasets/${route.params.id}/versions/${vid}/episodes/${route.params.ep}`);
});
function onTime() {
  t.value = video.value?.currentTime || 0;
}
</script>
<template>
  <div>
    <h2>Episode {{ route.params.ep }}</h2>
    <p class="muted" v-if="ep?.missing">{{ ep.missing }}</p>
    <video v-else-if="ep?.media_url" ref="video" controls style="width:100%;max-width:720px" :src="ep.media_url" @timeupdate="onTime" />
    <p v-else class="muted">缺媒体 — 没有视频文件，不会用纯色 canvas 冒充播放器。</p>
    <Chart title="action" :points="ep?.action_series || []" />
    <p class="muted">currentTime {{ t.toFixed(2) }}s</p>
  </div>
</template>
