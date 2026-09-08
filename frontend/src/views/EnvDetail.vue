<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";
import ObsStrip from "../components/ObsStrip.vue";

const route = useRoute();
const router = useRouter();
const data = ref<any>(null);
const refs = ref<any>(null);
const yamlOpen = ref(false);
const yamlText = ref("");
const err = ref("");
const msg = ref("");

onMounted(load);
watch(() => route.params.eeId, load);

async function load() {
  const id = route.params.eeId as string;
  data.value = await api(`/embodiment-envs/${id}`);
  refs.value = await api(`/embodiment-envs/${id}/refs`);
  yamlText.value = JSON.stringify(data.value.spec, null, 2);
  err.value = "";
  msg.value = "";
}

const shortId = computed(() => {
  const ids: string[] = refs.value?.env_ids || [];
  return ids.map((id) => ({
    id,
    label: id.replace("ee_bhl_", "").replace(/_v\d+$/, ""),
  }));
});

const draft = computed(() => data.value?.status === "draft");

function cellClass(status: string) {
  if (status === "ok") return "gate-ok";
  if (status === "fail") return "gate-fail";
  if (status === "warn") return "gate-warn";
  return "gate-none";
}

function copyId() {
  if (data.value?.id) navigator.clipboard?.writeText(data.value.id);
}

async function saveDraft() {
  err.value = "";
  try {
    const spec = JSON.parse(yamlText.value);
    await api(`/embodiment-envs/${data.value.id}`, { method: "PATCH", body: JSON.stringify({ spec }) });
    msg.value = "草稿已保存（只写平台，未写上游仓）";
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}

async function publish() {
  err.value = "";
  try {
    await api(`/embodiment-envs/${data.value.id}/publish`, { method: "POST", body: "{}" });
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}

async function cloneEnv() {
  const row = await api(`/embodiment-envs/${data.value.id}/clone`, { method: "POST", body: "{}" });
  router.push(`/embodiments/${row.embodiment_id}/envs/${row.id}`);
}

async function deprecate() {
  await api(`/embodiment-envs/${data.value.id}/deprecate`, { method: "POST", body: "{}" });
  await load();
}

async function remove() {
  err.value = "";
  try {
    const embId = data.value.embodiment_id;
    await api(`/embodiment-envs/${data.value.id}`, { method: "DELETE" });
    router.push(`/embodiments/${embId}`);
  } catch (e: any) {
    err.value = e.message;
  }
}

async function downloadYaml() {
  const r = await fetch(`/api/v1/embodiment-envs/${data.value.id}/yaml`, { credentials: "include" });
  const text = await r.text();
  const blob = new Blob([text], { type: "text/yaml" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${data.value.id}.yaml`;
  a.click();
}
</script>
<template>
  <div v-if="data">
    <div class="page-head">
      <div>
        <p class="crumb"><router-link :to="`/embodiments/${data.embodiment_id}`">本体</router-link> / {{ data.id }}</p>
        <h2>
          {{ data.morphology }}
          <span class="badge">{{ data.status }}</span>
          <span class="badge">{{ data.self_check?.passed ? "自洽通过" : "自洽失败" }}</span>
        </h2>
      </div>
      <div class="row">
        <button type="button" @click="copyId">复制 id</button>
        <button type="button" @click="downloadYaml">下载 YAML</button>
        <button type="button" v-if="draft" class="primary" @click="saveDraft">保存草稿</button>
        <button type="button" v-if="draft" class="primary" @click="publish">发布</button>
        <button type="button" v-if="draft" @click="remove">删除草稿</button>
        <button type="button" v-if="!draft" @click="cloneEnv">克隆为 _vN</button>
        <button type="button" v-if="data.status === 'active'" @click="deprecate">废弃</button>
      </div>
    </div>
    <p class="error">{{ err }}</p>
    <p class="ok xs" v-if="msg">{{ msg }}</p>

    <div class="kpis">
      <div class="card"><div class="l">obs</div><div class="v">{{ data.concat === false ? data.obs_dim + "+img" : data.obs_dim }}</div><div class="muted">history × {{ data.history_length }}</div></div>
      <div class="card"><div class="l">act / dof</div><div class="v">{{ data.action_dim }} / {{ data.dof }}</div><div class="muted">{{ data.action_type || "position" }} {{ data.action_scale != null ? "· scale " + data.action_scale : "" }} {{ data.action_absolute ? "· 绝对" : "" }}</div></div>
      <div class="card"><div class="l">policy</div><div class="v">{{ data.policy_hz }} <small>Hz</small></div><div class="muted">dt {{ data.frequency?.sim_dt }} × dec {{ data.frequency?.decimation }}</div></div>
      <div class="card"><div class="l">上机延迟上限</div><div class="v">{{ data.frequency?.max_policy_latency_ms }} <small>ms</small></div></div>
    </div>

    <div class="panel">
      <h3>观测拼接顺序 <span class="muted">从左到右 = 张量 index；concat={{ data.concat }}</span></h3>
      <ObsStrip v-if="data.layout?.length" :items="data.layout" />
      <p class="muted xs" v-if="data.concat === false">不要画成一条长向量。下轨是独立模态。</p>
      <div v-for="m in data.modalities || []" :key="m.name" class="obs-strip" style="margin-top:.5rem;height:28px;line-height:28px;font-size:11px">
        <span :style="{ flex: 1, background: '#5b8cff' }">{{ m.name }} {{ (m.shape || []).join("×") }} {{ m.dtype }}　不拼接</span>
      </div>
    </div>

    <div class="panel" v-if="data.action_terms?.length > 1 || data.action_terms?.[0]?.name !== 'all'">
      <h3>动作 terms</h3>
      <div class="obs-strip" style="height:22px;font-size:10px;line-height:22px">
        <span v-for="t in data.action_terms" :key="t.name" :style="{ flex: t.dim, background: t.name === 'arm' ? '#ff7ac0' : '#ff884d' }">
          {{ t.name }} {{ t.dim }} {{ t.type }}
        </span>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">
        <h3>joint_order（权威顺序）</h3>
        <div class="joint-grid">
          <div v-for="g in data.joint_groups || []" :key="g.name">
            <b>{{ String(g.name).toUpperCase() }} · {{ g.count }} · idx {{ g.index_start }}–{{ g.index_start + g.count - 1 }}</b><br>
            <template v-for="(j, i) in g.joints" :key="j">
              <span class="idx">{{ g.index_start + i }}</span>{{ j }}<br>
            </template>
          </div>
        </div>
      </div>
      <div>
        <div class="panel">
          <h3>频率自洽</h3>
          <dl class="dl">
            <dt>sim_dt</dt><dd>{{ data.frequency?.sim_dt }} s</dd>
            <dt>decimation</dt><dd>{{ data.frequency?.decimation }}</dd>
            <dt>1/(dt×dec)</dt><dd :class="data.frequency?.ok ? 'ok' : 'bad'">{{ data.frequency?.derived_hz }} Hz {{ data.frequency?.ok ? '= policy_hz' : '≠ policy_hz' }}</dd>
            <dt>control_hz</dt><dd>{{ data.frequency?.control_hz }}</dd>
            <dt>command_dim</dt><dd>{{ data.frequency?.command_dim ?? "—" }}</dd>
          </dl>
        </div>
        <div class="panel">
          <h3>绑定（只读）</h3>
          <dl class="dl">
            <dt>isaac_task</dt><dd>{{ data.sim_bindings?.isaac_task }}</dd>
            <dt>experiment</dt><dd>{{ data.sim_bindings?.experiment_name }}</dd>
            <dt>UDP</dt><dd>{{ data.real_bindings?.udp ? Object.values(data.real_bindings.udp).join(" / ") : "—" }}</dd>
            <dt>firmware_n_joints</dt><dd>{{ data.real_bindings?.firmware_n_joints }}</dd>
          </dl>
        </div>
      </div>
    </div>

    <div class="panel">
      <h3>自洽断言</h3>
      <table>
        <thead><tr><th>断言</th><th>expected</th><th>actual</th><th>结果</th></tr></thead>
        <tbody>
          <tr v-for="a in data.assertions || []" :key="a.name">
            <td class="mono xs">{{ a.name }}</td>
            <td>{{ a.expected }}</td>
            <td>{{ a.actual }}</td>
            <td :class="a.passed ? 'ok' : 'bad'">{{ a.passed ? "通过" : "失败" }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <h3>G1–G5 引用矩阵 <span class="muted">实测 shape；没有产物显示 — / 未注册</span></h3>
      <table v-if="refs?.matrix">
        <thead>
          <tr>
            <th>引用方</th>
            <th v-for="c in shortId" :key="c.id">{{ c.label }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in refs.matrix" :key="row.gate">
            <td>{{ row.title }}</td>
            <td v-for="c in shortId" :key="c.id" :class="cellClass(row.cells[c.id]?.status)">
              {{ row.cells[c.id]?.label }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <h3>引用方（本本体环境）</h3>
      <p class="xs">Jobs {{ refs?.jobs?.length || 0 }} · Models {{ refs?.models?.length || 0 }} · Datasets {{ refs?.datasets?.length || 0 }}</p>
      <ul class="xs">
        <li v-for="j in refs?.jobs || []" :key="j.id"><router-link to="/train/queue">{{ j.id }}</router-link> {{ j.kind }} {{ j.status }}</li>
        <li v-for="m in refs?.models || []" :key="m.id"><router-link :to="`/models/${m.id}`">{{ m.name }}</router-link></li>
      </ul>
    </div>

    <div class="panel">
      <h3>YAML <span class="lock" v-if="!draft">已发布只读</span><span class="badge" v-else>草稿可改</span></h3>
      <p class="muted xs">spec_hash {{ data.spec_hash }}。已发布不可改数字，要演进请克隆 _vN。</p>
      <button type="button" @click="yamlOpen = !yamlOpen">{{ yamlOpen ? "收起" : "展开" }}</button>
      <textarea v-if="yamlOpen && draft" v-model="yamlText" style="width:100%;min-height:280px;margin-top:.6rem;font:12px/1.45 ui-monospace,monospace" />
      <pre v-else-if="yamlOpen">{{ JSON.stringify(data.spec, null, 2) }}</pre>
    </div>
  </div>
</template>
