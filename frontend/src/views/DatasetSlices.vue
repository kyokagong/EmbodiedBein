<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import DataTabs from "../components/DataTabs.vue";

const route = useRoute();
const router = useRouter();
const ds = ref<any>(null);
const query = ref({ task: "", success: "", dur_min: "4", dur_max: "10", embodiment_env_id: "", saturation: "< 5%" });
const preview = ref<any>(null);
const err = ref("");
const job = ref<any>(null);

onMounted(async () => {
  ds.value = await api(`/datasets/${route.params.id}`);
  query.value.embodiment_env_id = ds.value.embodiment_env_id || "";
  await runPreview();
});

async function runPreview() {
  preview.value = await api(`/datasets/${route.params.id}/slice/preview`, {
    method: "POST",
    body: JSON.stringify(query.value),
  });
}

async function submit() {
  err.value = "";
  try {
    job.value = await api("/jobs", {
      method: "POST",
      body: JSON.stringify({
        spec: {
          kind: "data.slice",
          embodiment_env_id: query.value.embodiment_env_id || "ee_bhl_fixedhand_10dof_v1",
          params: { dataset_id: route.params.id, query: query.value, content_hash: ds.value?.versions?.[0]?.content_hash },
        },
      }),
    });
    if (job.value?.id) router.push("/train/queue");
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">数据 / {{ route.params.id }}</p>
        <h2>切片构造器</h2>
      </div>
    </div>
    <DataTabs :id="String(route.params.id)" />
    <p class="muted">切片是不可变 DatasetVersion。先预览命中数，再提交 data.slice。没有 episode 索引时命中数为空，不用占位 1128。</p>
    <div class="grid2">
      <div class="panel">
        <h3>过滤器（AND）</h3>
        <div class="form-grid">
          <span>task</span><input v-model="query.task" placeholder="reach" />
          <span>成功</span><input v-model="query.success" placeholder="true" />
          <span>时长 (s)</span><span class="row"><input v-model="query.dur_min" style="width:5rem" /> – <input v-model="query.dur_max" style="width:5rem" /></span>
          <span>环境 G2</span><input v-model="query.embodiment_env_id" />
          <span>饱和帧</span><input v-model="query.saturation" />
        </div>
        <div class="row" style="margin-top:.8rem">
          <button type="button" @click="runPreview">预览</button>
          <button type="button" class="primary" @click="submit">提交 data.slice</button>
        </div>
        <p class="error">{{ err }}</p>
      </div>
      <div class="panel">
        <h3>命中预览</h3>
        <p v-if="preview">
          命中 <b>{{ preview.hits == null ? "—" : preview.hits }}</b>
          / 文件 {{ preview.total }}
        </p>
        <p class="warn xs" v-if="preview?.warning">{{ preview.warning }}</p>
        <ul class="xs">
          <li v-for="(s, i) in preview?.preview || []" :key="i">{{ s.path || s.task || JSON.stringify(s) }}</li>
        </ul>
      </div>
    </div>
  </div>
</template>
