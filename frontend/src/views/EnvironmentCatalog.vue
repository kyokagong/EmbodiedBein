<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api/client";

const route = useRoute();
const router = useRouter();
const environments = ref<any[]>([]);
const templates = ref<any[]>([]);
const suites = ref<any[]>([]);
const filter = ref("all");
const error = ref("");
const loaded = ref(false);
const busy = ref(false);
const draft = ref<any>(null);
const selected = computed(() => environments.value.find(e => e.id === route.params.id));
const visible = computed(() => environments.value.filter(e => filter.value === "all" || e.usage === filter.value));
const bindings = computed(() => selected.value?.usage === "train"
  ? templates.value.filter(t => t.ready !== false && t.embodiment_env_id === selected.value.embodiment_env_id)
  : suites.value.filter(s => s.frozen && s.embodiment_env_id === selected.value?.embodiment_env_id &&
      (['rpo_wbc', 'robotwin'].includes(s.spec?.benchmark) || ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'].includes(s.spec?.suite))));
const settingFields = computed(() => selected.value?.usage === "train" && draft.value?.template_id === "rpo_v2_wbc_stand_walk_ppo"
  ? [["target_speed_mps", "前向目标速度 m/s"], ["stand_fraction", "站立场景比例"],
     ["velocity_reward_weight", "速度跟踪奖励权重"], ["upright_reward_weight", "直立奖励权重"]]
  : selected.value?.usage === "eval" && draft.value?.suite_id?.endsWith("_stand_walk_v2")
  ? [["stand_max_drift_m", "站立最大漂移 m"], ["walk_min_forward_m", "直行最小前进 m"]] : []);

async function load() {
  try {
    const [envs, ts, ss] = await Promise.all([api("/environments"), api("/train/templates"), api("/evals/suites")]);
    environments.value = envs.items || [];
    templates.value = ts.items || [];
    suites.value = ss.items || [];
  } catch (e: any) { error.value = e.message; }
  finally { loaded.value = true; }
}
watch(selected, row => { draft.value = row ? structuredClone(row) : null; error.value = ""; }, { immediate: true });
onMounted(load);

async function clone() {
  if (!selected.value || busy.value) return;
  busy.value = true; error.value = "";
  try {
    const row = await api(`/environments/${selected.value.id}/clone`, { method: "POST", body: "{}" });
    await load();
    await router.push(`/environments/${row.id}`);
  } catch (e: any) { error.value = e.message; }
  finally { busy.value = false; }
}
async function save(publish = false) {
  if (!draft.value || busy.value) return;
  busy.value = true; error.value = "";
  try {
    await api(`/environments/${draft.value.id}`, { method: "PATCH", body: JSON.stringify({
      name: draft.value.name, settings: draft.value.definition.settings,
      template_id: draft.value.template_id, suite_id: draft.value.suite_id,
    }) });
    if (publish) await api(`/environments/${draft.value.id}/publish`, { method: "POST", body: "{}" });
    await load();
  } catch (e: any) { error.value = e.message; }
  finally { busy.value = false; }
}
</script>
<template>
  <div>
    <div class="page-head"><div><p class="crumb">环境</p><h2>训练与评估环境</h2></div><router-link to="/envs">运行环境、镜像与 Doctor →</router-link></div>
    <p class="muted">环境定义任务、奖励与判据；本体接口定义观测和动作。已发布环境需创建修订后编辑。</p>
    <p class="error" role="alert">{{ error }}</p>
    <p v-if="loaded && route.params.id && !selected" class="error">所选环境不存在。</p>
    <div v-if="!route.params.id" class="row" aria-label="环境用途筛选"><button v-for="item in [['all','全部'],['train','训练'],['eval','评估']]" :key="item[0]" type="button" :class="{ primary: filter === item[0] }" @click="filter = item[0]">{{ item[1] }}</button></div>
    <div v-if="!route.params.id" class="panel">
      <table><thead><tr><th>环境</th><th>用途</th><th>任务</th><th>奖励</th><th>失败条件</th><th>成功条件</th><th>状态</th></tr></thead>
        <tbody><tr v-for="row in visible" :key="row.id">
          <td><router-link :to="`/environments/${row.id}`">{{ row.name }}</router-link><div class="muted xs">{{ row.id }}</div></td>
          <td>{{ row.usage === 'train' ? '训练' : '评估' }}</td><td>{{ row.definition.task }}</td><td>{{ row.definition.reward }}<div v-if="Object.keys(row.definition.settings || {}).length" class="muted xs">生效参数：{{ JSON.stringify(row.definition.settings) }}</div></td>
          <td>{{ row.definition.failure }}</td><td>{{ row.definition.success }}</td><td>{{ row.status }}</td>
        </tr></tbody>
      </table>
      <p v-if="!visible.length" class="muted">暂无环境。</p>
    </div>
    <div v-if="selected && draft" class="panel">
      <router-link to="/environments">← 返回环境列表</router-link>
      <div class="page-head"><div><h3>{{ selected.name }} <span class="badge">{{ selected.status }}</span></h3><p class="muted xs">本体接口 {{ selected.embodiment_env_id }} · 任务 {{ selected.task_id }} · {{ selected.content_hash }}</p></div>
        <div class="row"><button v-if="selected.status !== 'draft'" type="button" :disabled="busy" @click="clone">创建修订</button>
          <router-link :to="selected.usage === 'train' ? `/train/new?environment_id=${selected.id}` : `/eval/new?environment_id=${selected.id}`" v-if="selected.status === 'active'">用于{{ selected.usage === 'train' ? '训练' : '评估' }} →</router-link></div>
      </div>
      <p v-if="selected.revision_of" class="muted xs">修订自 <router-link :to="`/environments/${selected.revision_of}`">{{ selected.revision_of }}</router-link></p>
      <p v-for="child in environments.filter(e => e.revision_of === selected.id)" :key="child.id" class="muted xs">后续修订 <router-link :to="`/environments/${child.id}`">{{ child.name }} · {{ child.status }}</router-link></p>
      <fieldset :disabled="selected.status !== 'draft' || busy" style="border:0;padding:0;min-width:0">
        <label class="form-label">名称</label><input class="form-input" v-model="draft.name" maxlength="120" />
        <label class="form-label">绑定{{ selected.usage === 'train' ? '模板 / 任务' : '套件 / 任务' }}</label>
        <select v-if="selected.usage === 'train'" class="form-select" v-model="draft.template_id" @change="draft.definition.settings = {}"><option v-for="t in bindings" :key="t.id" :value="t.id">{{ t.label || t.id }}</option></select>
        <select v-else class="form-select" v-model="draft.suite_id" @change="draft.definition.settings = {}"><option v-for="s in bindings" :key="s.id" :value="s.id">{{ s.name }}</option></select>
        <p class="muted xs">切换绑定会改变后续作业的实际任务；必须保持本体接口兼容。</p>
        <dl class="dl"><template v-for="field in [['task','任务'],['reward','奖励'],['failure','失败条件'],['success','成功条件']]" :key="field[0]">
          <dt>{{ field[1] }}</dt><dd>{{ selected.definition[field[0]] }}</dd>
        </template></dl>
        <p class="muted xs">上述说明由绑定与生效参数生成；修改下列参数并保存后会刷新说明。</p>
        <div v-if="settingFields.length" class="grid2">
          <div v-for="[key, label] in settingFields" :key="key"><label class="form-label" :for="`env-setting-${key}`">{{ label }}</label>
            <input :id="`env-setting-${key}`" class="form-input" type="number" step="any" v-model.number="draft.definition.settings[key]" /></div>
        </div>
        <p v-else class="muted xs">此入口的执行参数由固定 recipe / 套件控制，没有可编辑数值。</p>
        <div v-if="selected.status === 'draft'" class="row"><button type="button" :disabled="busy" @click="save(false)">保存草稿</button><button type="button" class="primary" :disabled="busy" @click="save(true)">保存并发布</button></div>
      </fieldset>
    </div>
  </div>
</template>
