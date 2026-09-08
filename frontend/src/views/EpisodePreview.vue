<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import Chart from "../components/Chart.vue";
import DataTabs from "../components/DataTabs.vue";

const route = useRoute();
const ep = ref<any>(null);
const ds = ref<any>(null);
const video = ref<HTMLVideoElement | null>(null);
const t = ref(0);

onMounted(load);
watch(() => [route.params.id, route.params.ep], load);

async function load() {
  ds.value = await api(`/datasets/${route.params.id}`);
  const vid = ds.value.versions?.[0]?.id;
  const epid = String(route.params.ep);
  if (vid && epid && epid !== "missing" && epid !== "_") {
    try {
      ep.value = await api(`/datasets/${route.params.id}/versions/${vid}/episodes/${epid}`);
    } catch {
      ep.value = { missing: "未找到该 Episode 媒体或轨迹", action_series: [], media_url: null };
    }
  } else {
    ep.value = { missing: "缺媒体", action_series: [], media_url: null };
  }
}

function onTime() {
  t.value = video.value?.currentTime || 0;
}

function epLink(s: any) {
  const name = (s.path || "").split("/").pop() || "";
  const stem = name.replace(/\.[^.]+$/, "");
  return `/data/${route.params.id}/episodes/${encodeURIComponent(stem)}`;
}
</script>

<template>
  <div class="episode-preview-page">
    <div class="page-head">
      <div>
        <p class="crumb"><router-link :to="`/data/${route.params.id}`">{{ ds?.name || route.params.id }}</router-link> / Episode 预览</p>
        <h2>Episode: <span class="mono text-cyan">{{ route.params.ep }}</span></h2>
      </div>
      <div class="row">
        <span class="badge b-cyan">{{ ds?.format || 'format' }}</span>
        <span class="badge b-muted">环境: {{ ds?.embodiment_env_id || 'ee_*' }}</span>
      </div>
    </div>
    <DataTabs :id="String(route.params.id)" />

    <!-- 视频与轨迹双轴同步区 -->
    <div class="grid2" style="margin-top:1rem">
      <!-- 视频播放区 -->
      <div class="panel">
        <h3>视频画面 (Video Stream)</h3>
        <div class="video-container" v-if="ep?.media_url">
          <video
            ref="video"
            controls
            style="width:100%; border-radius:8px; background:#000;"
            :src="ep.media_url"
            @timeupdate="onTime"
          />
          <div class="playhead-info">
            <span>Playhead 时间戳: <b class="mono">{{ t.toFixed(2) }}s</b></span>
            <span class="muted xs">拖动视频进度条可同步右侧关节轨迹指针</span>
          </div>
        </div>
        <div class="empty-box" v-else>
          <b>缺媒体流文件</b>
          <p class="muted small mb0">
            {{ ep?.missing || '当前数据集切片无独立 MP4 视频，仅包含数值轨迹。' }}
          </p>
        </div>
      </div>

      <!-- 关节与动作曲线同步 -->
      <div class="panel">
        <h3>动作与状态曲线 (Action / State Timeline)</h3>
        <div v-if="ep?.action_series?.length">
          <Chart title="10 维动作轨迹 (Action Jitter & Position)" :points="ep.action_series" :playhead="t" />
        </div>
        <div class="empty-box" v-else>
          <b>无并列关节轨迹</b>
          <p class="muted small mb0">
            该原始切片未完成重定向 (Retargeting)；无 10 维机器人关节动作，系统不绘制虚假零线。
          </p>
        </div>

        <!-- 文本叙述时间轴 -->
        <h4 style="margin-top:1.2rem">语言指令与叙述 (Narration)</h4>
        <div class="narration-box">
          <div class="mono small font-bold">Goal Text: "pick up the green bottle and place it on tray"</div>
          <div class="muted xs">时间戳: 0.00s ~ {{ t.toFixed(2) }}s · 动作标签: [reach, grasp, move, place]</div>
        </div>
      </div>
    </div>

    <!-- 样本切换卡片 -->
    <div class="panel" style="margin-top:1rem" v-if="ds?.versions?.[0]?.stats?.samples?.length">
      <h3>同版本其他样本列表 (Samples)</h3>
      <div class="sample-grid">
        <router-link
          v-for="s in ds.versions[0].stats.samples.slice(0, 16)"
          :key="s.path"
          :to="epLink(s)"
          class="sample-chip"
          :class="{ active: (s.path || '').includes(String(route.params.ep)) }"
        >
          {{ (s.path || '').split('/').pop() }}
        </router-link>
      </div>
    </div>
  </div>
</template>

<style scoped>
.episode-preview-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.text-cyan { color: var(--cyan, #53d4ff); }
.playhead-info { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; font-size: 0.85rem; color: var(--muted, #9fb4c2); }
.empty-box { background: #050c12; border: 1px dashed var(--line, #284052); border-radius: 8px; padding: 1.5rem; text-align: center; color: var(--muted, #9fb4c2); }
.mb0 { margin-bottom: 0; }
.narration-box { background: #050c12; border: 1px solid var(--line, #284052); border-radius: 6px; padding: 0.8rem; margin-top: 0.5rem; }
.sample-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
.sample-chip { padding: 4px 10px; background: #050c12; border: 1px solid var(--line, #284052); border-radius: 6px; font-size: 0.8rem; font-family: var(--mono, monospace); text-decoration: none; color: var(--text, #edf6fb); }
.sample-chip.active { background: rgba(83, 212, 255, 0.15); border-color: var(--cyan); color: var(--cyan); font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.b-muted { background: rgba(255, 255, 255, 0.08); color: #9fb4c2; }
</style>
