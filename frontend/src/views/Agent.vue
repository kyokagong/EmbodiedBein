<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api/client";

const router = useRouter();
const sid = ref("");
const text = ref("");
const log = ref<{ role: string; text: string; guardrail?: string }[]>([]);
const sending = ref(false);
const quotas = ref<any>(null);

const quickPrompts = [
  { label: "📋 每日简报", query: "生成今日训练与评估简报" },
  { label: "⚠️ 训练失败诊断", query: "诊断最近失败的训练任务" },
  { label: "🧹 资源与存储巡检", query: "巡检 GPU 与 CAS 存储资源" },
  { label: "🛡️ 上机放行预检", query: "检查策略包是否满足上机放行要求" },
  { label: "🌙 夜间队列状态", query: "查看夜间待办队列" },
  { label: "📊 数据缺口分析", query: "分析数据漏斗与示教缺口" },
];

onMounted(async () => {
  try {
    const s = await api("/agent/sessions", { method: "POST", body: "{}" });
    sid.value = s.id;
  } catch {}
  try {
    quotas.value = await api("/system/quotas");
  } catch {}

  // Welcome message
  log.value.push({
    role: "agent",
    text: "你好！我是 EmbodiedBein 平台值班助手（规则优先 · 零后门 · 严格走 ToolBus）。\n你可以随时询问我训练状态、诊断失败任务或执行预置剧本。请点击下方快捷指令或直接输入：",
  });
});

async function sendQuery(q?: string) {
  const queryText = q || text.value;
  if (!queryText.trim() || sending.value) return;

  log.value.push({ role: "user", text: queryText });
  if (!q) text.value = "";
  sending.value = true;

  try {
    const r = await api(`/agent/sessions/${sid.value}/messages`, {
      method: "POST",
      body: JSON.stringify({ text: queryText }),
    });
    log.value.push({
      role: "agent",
      text: r.reply || "（无回复）",
      guardrail: r.guardrail,
    });
  } catch (e: any) {
    log.value.push({
      role: "agent",
      text: `❌ 请求失败: ${e.message}`,
    });
  } finally {
    sending.value = false;
  }
}

function renderMessage(content: string) {
  // Replace [ref#id] with clickable badges or spans
  return content;
}
</script>

<template>
  <div class="agent-page">
    <div class="page-head">
      <div>
        <p class="crumb">平台 Agent / 值班控制台</p>
        <h2>Agent 控制台 <span class="badge b-cyan">ToolBus 保护中</span></h2>
      </div>
      <router-link to="/agent/audit" class="btn-link">🛡️ 查看安全审计日志</router-link>
    </div>
    <p class="muted small">
      安全原则：Agent 没有任何后门，所有操作均通过 ToolBus 权限矩阵。<b>永久禁止</b>：真机放行 / 改已发布本体环境 / 门禁豁免。关掉 LLM 时规则路径 100% 可用。
    </p>

    <!-- 预算与状态条 -->
    <div class="kpis" v-if="quotas?.agent_gpu_hours">
      <div class="card">
        <div class="l">周 GPU·h 预算限额</div>
        <div class="v">{{ quotas.agent_gpu_hours.weekly_limit }} <small>GPU·h</small></div>
        <div class="muted">周硬顶保护</div>
      </div>
      <div class="card">
        <div class="l">本周已消耗</div>
        <div class="v">{{ quotas.agent_gpu_hours.used }} <small>GPU·h</small></div>
        <div class="muted">剩余 {{ quotas.agent_gpu_hours.remaining }} GPU·h</div>
      </div>
      <div class="card">
        <div class="l">决策模式</div>
        <div class="v" style="color:#53e5a3">规则优先 (llm=0)</div>
        <div class="muted">无外部网络依赖</div>
      </div>
      <div class="card">
        <div class="l">拦截动作数</div>
        <div class="v">0 次</div>
        <div class="muted">安全护栏实时生效</div>
      </div>
    </div>

    <!-- 快捷剧本药丸按钮 -->
    <div class="quick-prompts">
      <span class="muted small">快捷剧本：</span>
      <button
        v-for="p in quickPrompts"
        :key="p.label"
        type="button"
        class="prompt-pill"
        :disabled="sending"
        @click="sendQuery(p.query)"
      >
        {{ p.label }}
      </button>
    </div>

    <!-- 对话流窗口 -->
    <div class="chat-box">
      <div
        v-for="(m, i) in log"
        :key="i"
        class="msg-bubble"
        :class="m.role === 'user' ? 'user-msg' : 'agent-msg'"
      >
        <div class="msg-header">
          <span class="mono font-bold">{{ m.role === 'user' ? '👤 研发操作员' : '🤖 EmbodiedBein Agent' }}</span>
          <span class="badge b-amber" v-if="m.guardrail === 'blocked'">已触发安全拦截</span>
        </div>
        <div class="msg-body" style="white-space:pre-wrap; line-height:1.6;">{{ m.text }}</div>
      </div>
      <div v-if="sending" class="msg-bubble agent-msg">
        <div class="muted small">🤖 Agent 正在调用 ToolBus 工具并组装回复...</div>
      </div>
    </div>

    <!-- 底部输入框 -->
    <div class="input-bar">
      <input
        v-model="text"
        placeholder="输入指令（例如：帮我看看队列和显存、诊断失败任务）..."
        :disabled="sending"
        @keyup.enter="sendQuery()"
      />
      <button type="button" class="primary" :disabled="sending || !text.trim()" @click="sendQuery()">
        {{ sending ? "正在执行..." : "发送指令" }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.agent-page { max-width: 1100px; margin: 0 auto; padding: 1rem; }
.quick-prompts { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 1rem; }
.prompt-pill { background: #050c12; border: 1px solid var(--line, #284052); color: var(--text, #edf6fb); padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; cursor: pointer; }
.prompt-pill:hover:not(:disabled) { background: rgba(83, 212, 255, 0.15); border-color: var(--cyan, #53d4ff); color: var(--cyan); }
.chat-box { background: #050c12; border: 1px solid var(--line, #284052); border-radius: 12px; padding: 1.2rem; min-height: 380px; max-height: 520px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; margin-bottom: 1rem; }
.msg-bubble { padding: 0.8rem 1rem; border-radius: 8px; max-width: 85%; }
.user-msg { background: rgba(83, 212, 255, 0.1); border: 1px solid rgba(83, 212, 255, 0.25); align-self: flex-end; }
.agent-msg { background: var(--panel, #0d1924); border: 1px solid var(--line, #284052); align-self: flex-start; }
.msg-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.8rem; color: var(--muted, #9fb4c2); }
.font-bold { font-weight: 600; }
.input-bar { display: flex; gap: 10px; }
.input-bar input { flex: 1; padding: 10px 14px; background: #050c12; border: 1px solid var(--line, #284052); border-radius: 8px; color: #edf6fb; }
.btn-link { padding: 4px 12px; font-size: 0.85rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 6px; text-decoration: none; }
.primary { background: var(--cyan, #53d4ff); color: #071018; font-weight: 700; border: none; padding: 0 20px; border-radius: 8px; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }
.b-cyan { background: rgba(83, 212, 255, 0.15); color: #53d4ff; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
</style>
