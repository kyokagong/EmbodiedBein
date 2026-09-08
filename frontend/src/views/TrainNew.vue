<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const templates = ref<any[]>([]);
const embodiments = ref<any[]>([]);
const datasets = ref<any[]>([]);
const weights = ref<any[]>([]);
const images = ref<any[]>([]);
const err = ref("");
const preview = ref<any>(null);
const submitting = ref(false);

const form = ref({
  name: "bhl-wbc-ppo",
  template_id: "berkeley_wbc_ppo_mlp",
  embodiment_id: "",
  image: "",
  gpus: 2,
  gym_id: "",
  dataset_ids: [] as string[],
  base_model_id: "",
  max_epochs: 200,
  total_num_envs: 1024,
  lr: "1e-3",
  extra_text: "",
  auto_register_latest: true,
});

onMounted(async () => {
  templates.value = (await api("/train/templates")).items || [];
  embodiments.value = (await api("/embodiments")).items || [];
  if (embodiments.value[0]) form.value.embodiment_id = embodiments.value[0].id;
  const gyms = embodiments.value[0]?.gym_ids || [];
  if (gyms[0]) form.value.gym_id = gyms[0];
  datasets.value = (await api("/datasets")).items || [];
  try {
    weights.value = (await api("/weights")).items || [];
  } catch {
    weights.value = (await api("/models")).items || [];
  }
  try {
    images.value = (await api("/envs/images")).items || [];
    const def = images.value.find((i) => i.is_default);
    form.value.image = def?.name || images.value[0]?.name || "eb/rlinf:0.4.0-emb";
  } catch {
    form.value.image = "eb/rlinf:0.4.0-emb";
  }
  await refreshPreview();
});

watch(form, () => { refreshPreview(); }, { deep: true });

function extras(): string[] {
  return form.value.extra_text.split("\n").map((s) => s.trim()).filter(Boolean);
}

function payload() {
  return {
    name: form.value.name,
    template_id: form.value.template_id,
    embodiment_id: form.value.embodiment_id,
    image: form.value.image || null,
    gpus: Number(form.value.gpus),
    gym_id: form.value.gym_id || null,
    dataset_ids: form.value.dataset_ids.filter(Boolean),
    base_model_id: form.value.base_model_id || null,
    max_epochs: form.value.max_epochs || null,
    total_num_envs: form.value.total_num_envs || null,
    lr: form.value.lr || null,
    extra_overrides: extras(),
    auto_register_latest: form.value.auto_register_latest,
  };
}

async function refreshPreview() {
  if (!form.value.embodiment_id || !form.value.template_id) return;
  try {
    preview.value = await api("/train/preview", { method: "POST", body: JSON.stringify(payload()) });
    err.value = "";
  } catch (e: any) {
    preview.value = null;
    err.value = e.message;
  }
}

const selectedEmb = () => embodiments.value.find((e) => e.id === form.value.embodiment_id);

watch(() => form.value.embodiment_id, () => {
  const gyms = selectedEmb()?.gym_ids || [];
  if (gyms.length && !gyms.includes(form.value.gym_id)) form.value.gym_id = gyms[0] || "";
});

async function submit() {
  err.value = "";
  submitting.value = true;
  try {
    const row = await api("/train", { method: "POST", body: JSON.stringify(payload()) });
    router.push(`/train/${row.id}`);
  } catch (e: any) {
    err.value = e.message;
  } finally {
    submitting.value = false;
  }
}
</script>
<template>
  <div>
    <div class="page-head">
      <div>
        <p class="crumb">训练 / 新建</p>
        <h2>新建 RLinf 训练</h2>
      </div>
      <router-link class="badge" to="/train">← 返回列表</router-link>
    </div>
    <p class="muted xs">本版只提交 RLinf 模板。Isaac RSL-RL 不进此表单。</p>
    <p class="error">{{ err }}</p>
    <div class="grid2">
      <div class="panel">
        <h3>配置</h3>
        <div class="form-group" style="margin-bottom:.8rem">
          <label class="form-label">任务名称</label>
          <input class="form-input" v-model="form.name" />
        </div>
        <div class="form-group" style="margin-bottom:.8rem">
          <label class="form-label">模板</label>
          <select class="form-select" v-model="form.template_id">
            <option v-for="t in templates" :key="t.id" :value="t.id">{{ t.label || t.id }}</option>
          </select>
        </div>
        <div class="grid2">
          <div>
            <label class="form-label">GPU 数</label>
            <select class="form-select" v-model.number="form.gpus">
              <option :value="1">1 卡</option>
              <option :value="2">2 卡</option>
              <option :value="4">4 卡</option>
            </select>
          </div>
          <div>
            <label class="form-label">镜像</label>
            <select class="form-select" v-model="form.image">
              <option v-for="im in images" :key="im.id" :value="im.name">{{ im.name }}</option>
              <option v-if="!images.length" value="eb/rlinf:0.4.0-emb">eb/rlinf:0.4.0-emb</option>
            </select>
          </div>
        </div>
        <div class="grid2" style="margin-top:.8rem">
          <div>
            <label class="form-label">本体</label>
            <select class="form-select" v-model="form.embodiment_id">
              <option v-for="e in embodiments" :key="e.id" :value="e.id">{{ e.display_name }}</option>
            </select>
          </div>
          <div>
            <label class="form-label">Gym ID</label>
            <select class="form-select" v-model="form.gym_id">
              <option value="">（模板默认）</option>
              <option v-for="g in selectedEmb()?.gym_ids || []" :key="g" :value="g">{{ g }}</option>
            </select>
          </div>
        </div>
        <div class="grid2" style="margin-top:.8rem">
          <div>
            <label class="form-label">数据集（可选）</label>
            <select class="form-select" v-model="form.dataset_ids[0]">
              <option value="">PPO 可不挂数据</option>
              <option v-for="d in datasets" :key="d.id" :value="d.id">{{ d.name }}</option>
            </select>
          </div>
          <div>
            <label class="form-label">基座模型（可选）</label>
            <select class="form-select" v-model="form.base_model_id">
              <option value="">从零初始化</option>
              <option v-for="w in weights" :key="w.id" :value="w.id">{{ w.name }}</option>
            </select>
          </div>
        </div>
        <div class="grid2" style="margin-top:.8rem">
          <div>
            <label class="form-label">max_epochs</label>
            <input class="form-input" type="number" v-model.number="form.max_epochs" />
          </div>
          <div>
            <label class="form-label">total_num_envs</label>
            <input class="form-input" type="number" v-model.number="form.total_num_envs" />
          </div>
        </div>
        <div style="margin-top:.8rem">
          <label class="form-label">学习率 actor.optim.lr</label>
          <input class="form-input" v-model="form.lr" />
        </div>
        <div style="margin-top:.8rem">
          <label class="form-label"><input type="checkbox" v-model="form.auto_register_latest" /> 成功后自动登记 latest checkpoint</label>
        </div>
        <div style="margin-top:.8rem">
          <label class="form-label">高级 Hydra Overrides（一行一个）</label>
          <textarea class="form-textarea" v-model="form.extra_text" placeholder="runner.save_interval=20" />
        </div>
        <button class="primary" style="margin-top:1rem;width:100%" :disabled="submitting" @click="submit">提交训练</button>
      </div>
      <div class="panel">
        <h3>预览 · Overrides 与挂载</h3>
        <pre class="preview-box" v-if="preview">镜像 {{ preview.image }}

{{ (preview.overrides || []).join("\n") }}

挂载
{{ (preview.mounts || []).join("\n") }}</pre>
        <p v-else class="muted">填写表单后这里显示 Hydra 覆盖与只读挂载。</p>
      </div>
    </div>
  </div>
</template>
