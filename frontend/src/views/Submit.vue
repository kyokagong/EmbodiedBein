<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";
import GatePanel from "../components/GatePanel.vue";

const embodiments = ref<any[]>([]);
const tasks = ref<any[]>([]);
const images = ref<string[]>([]);
const embId = ref("");

const jobKinds = [
  { id: "train.rl", label: "RL 训练 (IsaacLab PPO)", defaultImage: "eb/isaac-lab:2.3.2" },
  { id: "train.sft", label: "VLA 模仿微调 (RLinf SFT)", defaultImage: "eb/rlinf:0.4.0-emb" },
  { id: "eval.sim", label: "IsaacLab 仿真评测", defaultImage: "eb/isaac-lab:2.3.2" },
  { id: "eval.sim2sim", label: "MuJoCo 跨域评测 (sim2sim)", defaultImage: "eb/mujoco:latest" },
  { id: "export.policy", label: "导出策略为 ONNX", defaultImage: "eb/isaac-lab:2.3.2" },
  { id: "data.collect", label: "脚本专家示教数据采集", defaultImage: "eb/isaac-lab:2.3.2" },
  { id: "data.convert", label: "转换数据集 (LeRobot v2)", defaultImage: "eb/tools:latest" },
  { id: "data.slice", label: "创建数据集切片", defaultImage: "eb/tools:latest" },
  { id: "bundle.build", label: "打包策略包 (.ebp)", defaultImage: "eb/tools:latest" },
];

const spec = ref({
  kind: "train.rl",
  embodiment_env_id: "",
  task_id: "",
  name: "biped-smoke",
  params: {
    model_name: "",
    task: "",
    num_envs: 64,
    max_iterations: 5,
    seed: 0,
    stack: "isaaclab_rslrl",
    config_name: "",
    num_episodes: 10,
    policy_artifact: "",
    output_format: "onnx",
  },
  resources: { gpus: 1, shm_gb: 2 },
  policy: { priority: 50, timeout_h: 24 },
  image: "eb/isaac-lab:2.3.2",
});

const previewCmd = ref("");
const estimate = ref<any>(null);
const gate = ref<any>(null);
const result = ref<any>(null);
const err = ref("");
const loading = ref(false);
const router = useRouter();

const currentEmb = computed(() => embodiments.value.find((e) => e.id === embId.value));
const publishedEnvs = computed(() => (currentEmb.value?.envs || []).filter((e: any) => e.status === "active"));
const compatibleTasks = computed(() =>
  tasks.value.filter(
    (t) => t.status === "active" && (t.compatible_env_ids || []).includes(spec.value.embodiment_env_id),
  ),
);
const stacks = computed(() => {
  const t = compatibleTasks.value.find((x) => x.id === spec.value.task_id);
  const keys = Object.keys(t?.stack_bindings || {});
  return keys.length ? keys : ["isaaclab_rslrl"];
});

onMounted(async () => {
  embodiments.value = (await api("/embodiments")).items || [];
  tasks.value = (await api("/tasks")).items || [];
  images.value = (await api("/system/images")).items || [];
  const bhl = embodiments.value.find((e) => e.id === "emb_berkeley_humanoid_lite") || embodiments.value[0];
  if (bhl) {
    embId.value = bhl.id;
    const env = (bhl.envs || []).find((e: any) => e.id === "ee_bhl_biped_12dof_v1") || (bhl.envs || []).find((e: any) => e.status === "active");
    if (env) spec.value.embodiment_env_id = env.id;
  }
  syncTask();
  updatePreview();
});

watch(() => spec.value.kind, (newKind) => {
  const k = jobKinds.find((x) => x.id === newKind);
  if (k) spec.value.image = k.defaultImage;
  if (newKind === "train.sft") {
    spec.value.params.stack = "rlinf";
    spec.value.resources.gpus = 4;
  } else if (newKind === "eval.sim2sim" || newKind.startsWith("data.") || newKind === "bundle.build") {
    spec.value.resources.gpus = 0;
  } else {
    spec.value.params.stack = "isaaclab_rslrl";
    spec.value.resources.gpus = 1;
  }
  updatePreview();
});

watch(() => spec.value.embodiment_env_id, () => {
  syncTask();
  updatePreview();
});
watch(() => spec.value.task_id, () => {
  syncGym();
  updatePreview();
});
watch(() => spec.value.params, updatePreview, { deep: true });
watch(() => spec.value.resources, updatePreview, { deep: true });

function syncTask() {
  const t = compatibleTasks.value[0];
  spec.value.task_id = t?.id || "";
  syncGym();
}

function syncGym() {
  const t = tasks.value.find((x) => x.id === spec.value.task_id);
  const stack = spec.value.params.stack || "isaaclab_rslrl";
  const gym = t?.stack_bindings?.[stack]?.gym_id || t?.stack_bindings?.isaaclab_rslrl?.gym_id || "";
  spec.value.params.task = gym;
  if (t?.stack_bindings?.rlinf?.config_name && stack === "rlinf") {
    spec.value.params.config_name = t.stack_bindings.rlinf.config_name;
  }
}

watch(embId, () => {
  const env = publishedEnvs.value[0];
  spec.value.embodiment_env_id = env?.id || "";
  syncTask();
});

async function updatePreview() {
  try {
    const res = await api("/jobs/preview-command", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
    previewCmd.value = res.command || "";
  } catch {
    previewCmd.value = "";
  }
}

async function est() {
  loading.value = true;
  err.value = "";
  try {
    estimate.value = await api("/capacity/estimate", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
    gate.value = await api(`/embodiment-envs/${spec.value.embodiment_env_id}/validate`, {
      method: "POST",
      body: JSON.stringify({ spec: spec.value }),
    });
  } catch (e: any) {
    err.value = e.message;
  } finally {
    loading.value = false;
  }
}

async function submit() {
  loading.value = true;
  err.value = "";
  try {
    result.value = await api("/jobs", { method: "POST", body: JSON.stringify({ spec: spec.value }) });
    if (result.value?.id) router.push("/train/queue");
  } catch (e: any) {
    err.value = e.message;
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="submit-page">
    <div class="page-head">
      <div>
        <p class="crumb">训练与执行 / 任务提交</p>
        <h2>提交任务 <span class="badge b-cyan">{{ spec.kind }}</span></h2>
      </div>
      <div class="row">
        <button type="button" @click="est" :disabled="loading">🛡️ G1 预检 / 容量估算</button>
        <button type="button" class="primary" @click="submit" :disabled="loading">🚀 正式提交</button>
      </div>
    </div>
    <p class="muted small">执行原则：已发布本体环境 × 任务 × 训练栈 × 容器镜像四元组。控制面绝不跑子进程，所有计算由 ebd 动态容器化下发。</p>
    <p class="error" v-if="err">{{ err }}</p>

    <div class="grid2">
      <!-- 左侧表单 -->
      <div class="panel">
        <h3>1. 任务类型与执行目标</h3>
        <div class="form-grid">
          <span>任务类型 (Kind)</span>
          <select v-model="spec.kind">
            <option v-for="k in jobKinds" :key="k.id" :value="k.id">{{ k.label }} ({{ k.id }})</option>
          </select>

          <span>本体 (Embodiment)</span>
          <select v-model="embId">
            <option v-for="e in embodiments" :key="e.id" :value="e.id">{{ e.display_name }} ({{ e.id }})</option>
          </select>

          <span>本体环境 (已发布)</span>
          <select v-model="spec.embodiment_env_id">
            <option v-for="e in publishedEnvs" :key="e.id" :value="e.id">{{ e.id }} · {{ e.obs_dim }}/{{ e.action_dim }} @{{ e.policy_hz }}Hz</option>
          </select>

          <span>任务定义 (Task)</span>
          <select v-model="spec.task_id">
            <option v-for="t in compatibleTasks" :key="t.id" :value="t.id">{{ t.name || t.id }} ({{ t.id }})</option>
          </select>

          <span>运行栈 (Stack)</span>
          <select v-model="spec.params.stack">
            <option v-for="s in stacks" :key="s" :value="s">{{ s }}</option>
          </select>

          <span>任务镜像 (Image)</span>
          <select v-model="spec.image">
            <option v-for="im in images" :key="im" :value="im">{{ im }}</option>
          </select>
        </div>

        <h3 style="margin-top:1.5rem">2. 超参数与规格配置</h3>
        <div class="form-grid">
          <span>任务名称</span>
          <input v-model="spec.name" />
          <template v-if="spec.kind.startsWith('train.')">
            <span>产出模型名称</span>
            <input v-model="spec.params.model_name" :placeholder="spec.name || '默认使用任务名称'" />
          </template>

          <!-- Dynamic fields for train.rl -->
          <template v-if="spec.kind === 'train.rl' || spec.kind === 'eval.sim'">
            <span>Gym Task ID</span>
            <input v-model="spec.params.task" placeholder="e.g. Velocity-Biped-v0" />

            <span>并行环境数 (num_envs)</span>
            <input type="number" v-model.number="spec.params.num_envs" />

            <span>迭代步数 (max_iterations)</span>
            <input type="number" v-model.number="spec.params.max_iterations" />

            <span>随机种子 (seed)</span>
            <input type="number" v-model.number="spec.params.seed" />
          </template>

          <!-- Dynamic fields for train.sft -->
          <template v-if="spec.kind === 'train.sft'">
            <span>RLinf Config Name</span>
            <input v-model="spec.params.config_name" placeholder="e.g. bhl_fixedhand_vla_sft" />

            <span>批处理步数 (max_steps)</span>
            <input type="number" v-model.number="spec.params.max_iterations" />
          </template>

          <!-- Dynamic fields for eval.sim2sim -->
          <template v-if="spec.kind === 'eval.sim2sim'">
            <span>评测回合数 (num_episodes)</span>
            <input type="number" v-model.number="spec.params.num_episodes" />
          </template>

          <!-- Resource config -->
          <span>GPU 卡数 (0~4)</span>
          <input type="number" min="0" max="4" v-model.number="spec.resources.gpus" />

          <span>共享内存 (shm_gb)</span>
          <input type="number" min="1" v-model.number="spec.resources.shm_gb" />

          <span>优先级 (0~100)</span>
          <input type="number" min="0" max="100" v-model.number="spec.policy.priority" />

          <span>超时限制 (小时)</span>
          <input type="number" min="1" v-model.number="spec.policy.timeout_h" />
        </div>
      </div>

      <!-- 右侧预览与预检反馈 -->
      <div class="panel">
        <h3>实时等价 Docker 命令预览</h3>
        <p class="muted small">Adapter 将 JobSpec 翻译为以下容器启动命令：</p>
        <pre class="cmd-box"><code>{{ previewCmd || "# 正在计算等价命令..." }}</code></pre>

        <h3 style="margin-top:1.2rem">显存与容量预估</h3>
        <div class="card" v-if="estimate">
          <div class="kpi-row">
            <div>
              <div class="l">容量结论</div>
              <div class="v" :class="estimate.verdict === 'fit' ? 'text-ok' : 'text-warn'">{{ estimate.verdict === 'fit' ? '✓ 允许准入 (Fit)' : '⚠️ 显存不足 (No Fit)' }}</div>
            </div>
            <div>
              <div class="l">单卡峰值显存</div>
              <div class="v">{{ estimate.per_gpu_peak_gb }} <small>GB / 24 GB</small></div>
            </div>
          </div>
          <p class="muted xs">{{ estimate.note || "按 L2 解析模型与历史指纹估算" }}</p>
        </div>
        <p class="muted small" v-else>点击上方「G1 预检 / 容量估算」进行前置检查。</p>

        <h3 style="margin-top:1.2rem">G1 门禁自洽判定</h3>
        <GatePanel :result="gate" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.submit-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.cmd-box { background: #050c12; border: 1px solid #1d3243; border-radius: 8px; padding: 0.8rem; font-family: var(--mono, monospace); font-size: 0.82rem; color: #53d4ff; overflow-x: auto; white-space: pre-wrap; word-break: break-all; max-height: 220px; }
.kpi-row { display: flex; gap: 24px; margin-bottom: 0.5rem; }
.kpi-row .l { font-size: 0.8rem; color: var(--muted, #9fb4c2); }
.kpi-row .v { font-size: 1.3rem; font-weight: 700; font-family: var(--mono, monospace); }
.text-ok { color: #53e5a3; }
.text-warn { color: #ff5d68; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-family: var(--mono, monospace); }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.primary { background: var(--cyan, #53d4ff); color: #071018; font-weight: 700; border: none; }
.primary:hover { background: #7ee0ff; }
</style>
