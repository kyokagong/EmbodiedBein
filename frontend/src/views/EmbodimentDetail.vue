<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import UrdfStage from "../components/UrdfStage.vue";

const route = useRoute();
const router = useRouter();
const data = ref<any>(null);
const preview = ref<any>(null);
const scan = ref<any>(null);
const err = ref("");
const displayName = ref("");
const tags = ref("");
const urdf = ref("");

onMounted(load);
watch(() => route.params.id, load);

async function load() {
  const id = route.params.id as string;
  data.value = await api(`/embodiments/${id}`);
  displayName.value = data.value.display_name;
  tags.value = (data.value.tags || []).join(", ");
  const variants = data.value.urdf_variants || [];
  if (!urdf.value) urdf.value = data.value.default_urdf || variants[0]?.path || "";
  await loadPreview();
}

async function loadPreview() {
  const id = route.params.id as string;
  const q = urdf.value ? `?urdf=${encodeURIComponent(urdf.value)}` : "";
  try {
    preview.value = await api(`/embodiments/${id}/preview${q}`);
  } catch {
    preview.value = { joints: [], tree: [], note: "无 URDF" };
  }
}

watch(urdf, loadPreview);

async function saveMeta() {
  err.value = "";
  try {
    await api(`/embodiments/${route.params.id}`, {
      method: "PATCH",
      body: JSON.stringify({
        display_name: displayName.value,
        tags: tags.value.split(",").map((s) => s.trim()).filter(Boolean),
      }),
    });
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}

async function doScan() {
  scan.value = await api(`/embodiments/${route.params.id}/scan`, { method: "POST", body: "{}" });
  await load();
}

async function archive() {
  await api(`/embodiments/${route.params.id}/archive`, { method: "POST", body: "{}" });
  await load();
}

async function remove() {
  err.value = "";
  try {
    await api(`/embodiments/${route.params.id}`, { method: "DELETE" });
    router.push("/embodiments");
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>
<template>
  <div v-if="data">
    <div class="page-head">
      <div>
        <p class="crumb">本体 / {{ data.id }}</p>
        <h2>{{ data.display_name }} <span class="badge">{{ data.status }}</span></h2>
      </div>
      <div class="row">
        <button type="button" @click="doScan">重新扫描</button>
        <button type="button" @click="archive">归档</button>
        <button type="button" :disabled="data.seed" @click="remove">删除</button>
      </div>
    </div>
    <p class="error">{{ err }}</p>
    <p class="muted xs" v-if="scan">扫描 wrote={{ scan.wrote }} · {{ scan.head || scan.note || "" }}</p>

    <div class="grid2">
      <div class="panel">
        <h3>3D · 多 URDF + 关节滑杆</h3>
        <select v-if="(data.urdf_variants || []).length" v-model="urdf" style="margin-bottom:.6rem;max-width:100%">
          <option v-for="v in data.urdf_variants" :key="v.path || v" :value="v.path || v">
            {{ v.label || v.path || v }}
          </option>
        </select>
        <p class="muted xs">{{ preview?.note }} · {{ preview?.urdf_path || "无 URDF" }}</p>
        <UrdfStage
          v-if="preview?.urdf_url"
          :key="String(route.params.id) + ':' + (preview.urdf_path || '')"
          :embodiment-id="String(route.params.id)"
          :preview="preview"
          :envs="data.envs || []"
        />
        <p v-else class="muted">没有可解析的 URDF。磁盘上扫到什么列什么。</p>
      </div>
      <div class="panel">
        <h3>来源与扫描</h3>
        <dl class="dl">
          <dt>path</dt><dd>{{ data.root_path || data.source?.path || "—" }}</dd>
          <dt>URDF</dt><dd>{{ (data.urdf_variants || []).length }}</dd>
          <dt>gym</dt><dd>{{ (data.gym_ids || []).join(", ") || "—" }}</dd>
          <dt>扫描</dt><dd>{{ data.scanned_at || "—" }}</dd>
        </dl>
        <div class="form-grid" style="margin-top:.8rem">
          <span>显示名</span><input v-model="displayName" />
          <span>标签</span><input v-model="tags" />
        </div>
        <button type="button" style="margin-top:.6rem" @click="saveMeta">保存元数据</button>
        <p class="muted xs">只改元数据，不写上游仓。</p>
        <div style="margin-top:1rem" v-if="(data.gym_ids || []).length">
          <h3>Gym IDs</h3>
          <p class="mono xs" v-for="g in data.gym_ids" :key="g">{{ g }}</p>
        </div>
      </div>
    </div>
  </div>
</template>
