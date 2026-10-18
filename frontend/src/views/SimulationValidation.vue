<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute } from "vue-router";
import { api } from "../api/client";
import NativeSimulationStage from "../components/NativeSimulationStage.vue";
import { defaultLiveCamera, liveId, liveStreamUrl, validateLiveState, type LiveState, type LiveIdentity, type LiveCamera } from "../utils/ufoLive";

const route = useRoute();
const prefix = "/ufo-native/live";
const capabilities = ref<any>(null);
const modelId = ref("");
const environmentId = ref("");
const duration = ref(300);
const seed = ref(71000);
const preview = ref<any>(null);
const job = ref<any>(typeof route.query.session_job === "string" ? { id: route.query.session_job, status: "queued" } : null);
const activeSessions = ref<any[]>([]);
const activeSessionsError = ref("");
const otherClosing = ref<string[]>([]);
const otherCloseErrors = ref<Record<string, string>>({});
const closing = ref(false);
const closeRequested = ref(false);
const closeError = ref("");
const closeNotice = ref("");
const leaving = ref(false);
const suspended = ref(false);
const state = ref<LiveState | null>(null);
const stage = ref<InstanceType<typeof NativeSimulationStage> | null>(null);
const pending = ref<{ id: string; action: string; task?: string } | null>(null);
const commandSending = ref(false);
const commandUncertain = ref(false);
const connected = ref(false);
const busy = ref(false);
const error = ref("");
const transportError = ref("");
const mediaErrors = ref<Record<string, boolean>>({});
const now = ref(performance.now());
const receivedAt = ref(0);
const controller = new AbortController();
let jobTimer: ReturnType<typeof setTimeout> | undefined;
let heartbeatTimer: ReturnType<typeof setTimeout> | undefined;
let clockTimer: ReturnType<typeof setTimeout> | undefined;
let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
let cameraTimer: ReturnType<typeof setTimeout> | undefined;
let commandTimer: ReturnType<typeof setTimeout> | undefined;
let socket: WebSocket | null = null;
let socketRun = "", reconnectAttempt = 0;
let camera = defaultLiveCamera(), cameraSeq = 0;
let jobLoading = false, revision = 0, sessionRevision = 0;
let submissionPromise: Promise<any> | null = null;
let submissionKey: string | null = null;
let closePromise: Promise<boolean> | null = null;
const confirmedClosed = new Set<string>();
const confirmedPendingKeys = new Set<string>();
const fallbackClosures = new Map<string, Promise<void>>();
const terminal = (value: any) => ["succeeded", "failed", "timeout", "cancelled", "infra_failed"].includes(value?.status);
const liveEnded = computed(() => ["finalizing", "finished", "error"].includes(state.value?.status || ""));
const frozen = computed(() => !!job.value && !terminal(job.value));
const model = computed(() => capabilities.value?.models?.find((item: any) => item.id === modelId.value));
const environments = computed<any[]>(() => capabilities.value?.environments?.filter((item: any) => item.embodiment_env_id === model.value?.embodiment_env_id) || []);
const activeTask = computed(() => state.value?.task || "");
const stale = computed(() => !connected.value || !state.value || now.value - receivedAt.value > 1500);
const canControl = computed(() => !closeRequested.value && !closing.value && !suspended.value && !leaving.value && frozen.value && !stale.value && !transportError.value && !pending.value && ["running", "paused", "fallen"].includes(state.value?.status || ""));
const otherSessions = computed(() => activeSessions.value.filter(item => item.id !== job.value?.id));
const targetSpeed = computed(() => capabilities.value?.tasks?.find((task: any) => task.id === activeTask.value)?.speed);
const statusLabels: Record<string, string> = { initializing: "初始化", running: "运行中", paused: "已暂停", fallen: "跌倒，等待重置", finalizing: "正在整理录像与证据", finished: "实时会话结束", error: "会话异常", queued: "排队", succeeded: "诊断产物已收集", failed: "作业失败", timeout: "作业超时", infra_failed: "运行环境失败", cancelled: "已取消" };
const identity = (): LiveIdentity => ({ run_id: job.value.run_id, model_id: job.value.model_id, payload_sha256: job.value.payload_sha256, plan_hash: job.value.plan_hash });

watch([modelId, environmentId, duration, seed], () => { revision++; preview.value = null; });
watch(modelId, () => { if (!environments.value.some(item => item.id === environmentId.value)) environmentId.value = environments.value[0]?.id || ""; });
onMounted(() => {
  window.addEventListener("pagehide", pagehide); window.addEventListener("pageshow", pageshow);
  tickClock(); void load();
});
onBeforeRouteLeave(leaveEnvironment);
onBeforeRouteUpdate(async to => {
  const nextId = typeof to.query.session_job === "string" ? to.query.session_job : "";
  if (nextId === (job.value?.id || "")) return true;
  if (!await leaveEnvironment()) return false;
  leaving.value = false; closeRequested.value = false; closeNotice.value = "";
  state.value = null; pending.value = null;
  sessionRevision++;
  job.value = nextId ? { id: nextId, status: "queued" } : null;
  if (nextId) void pollJob();
  return true;
});
onUnmounted(() => {
  pagehide();
  controller.abort(); clearTimeout(jobTimer); clearTimeout(clockTimer); closeStream();
  window.removeEventListener("pagehide", pagehide); window.removeEventListener("pageshow", pageshow);
});
function tickClock() { now.value = performance.now(); if (!controller.signal.aborted && !suspended.value) clockTimer = setTimeout(tickClock, 250); }
function body() { return { model_id: modelId.value, environment_id: environmentId.value, initial_task: "move-ego-0-0", duration_seconds: duration.value, seed: seed.value }; }
async function load() {
  const sessions = loadActiveSessions();
  error.value = "";
  try {
    const result = await api(`${prefix}/capabilities`, { signal: controller.signal });
    if (controller.signal.aborted) return;
    capabilities.value = result;
    const requested = typeof route.query.model_id === "string" ? route.query.model_id : "";
    modelId.value = result.models?.find((item: any) => item.id === requested)?.id || result.models?.[0]?.id || "";
    if (requested && requested !== modelId.value) error.value = "指定模型不支持原生实时仿真，请选择兼容的已导出模型。";
    environmentId.value = environments.value[0]?.id || "";
    if (job.value) await pollJob();
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { await sessions; }
}
async function loadActiveSessions() {
  try {
    const result = await api(`${prefix}/sessions`, { signal: controller.signal });
    if (!controller.signal.aborted) { activeSessions.value = (result.items || []).filter((item: any) => !confirmedClosed.has(item.id)); activeSessionsError.value = ""; }
  } catch (reason: any) { if (!controller.signal.aborted) activeSessionsError.value = `读取活动交互环境失败：${reason.message}`; }
}
function confirmClosed(id: string, result: any) {
  if (result?.id !== id || !terminal(result)) throw Error("服务端尚未确认交互环境已关闭，请重试。");
  confirmedClosed.add(id);
  activeSessions.value = activeSessions.value.filter(item => item.id !== id);
  if (job.value?.id === id) { sessionRevision++; job.value = result; pending.value = null; closeError.value = ""; closeNotice.value = "交互环境已关闭，计算资源已释放。"; }
}
function confirmPendingClosed(key: string, result: any) {
  if (result?.closed !== true || (result.job !== null && (!result.job?.id || !terminal(result.job)))) throw Error("服务端尚未确认创建中的交互环境已关闭，请重试。");
  confirmedPendingKeys.add(key);
  const currentSubmission = submissionKey === key;
  if (currentSubmission) { submissionKey = null; submissionPromise = null; busy.value = false; preview.value = null; }
  if (result.job) { if (currentSubmission) job.value = result.job; confirmClosed(result.job.id, result.job); }
  else if (currentSubmission || !job.value) { closeError.value = ""; closeNotice.value = "交互环境已关闭，未保留活动创建请求。"; }
}
function closeEnvironment(): Promise<boolean> {
  if (closePromise) return closePromise;
  closing.value = true; closeRequested.value = true; closeError.value = ""; closeNotice.value = "";
  sessionRevision++;
  clearTimeout(jobTimer); closeStream();
  closePromise = (async () => {
    if (submissionKey) {
      const key = submissionKey;
      try {
        confirmPendingClosed(key, await api(`${prefix}/sessions/close-pending`, { method: "POST", body: JSON.stringify({ idempotency_key: key }) }));
        return true;
      } catch (reason: any) { closeError.value = `关闭创建中的交互环境失败：${reason.message} 请重试；当前页面将保留。`; return false; }
    }
    const id = job.value?.id;
    if (!id) return true;
    if (confirmedClosed.has(id)) { closeNotice.value = "交互环境已关闭，计算资源已释放。"; return true; }
    try {
      const result = await api(`${prefix}/sessions/${encodeURIComponent(id)}/close`, { method: "POST" });
      confirmClosed(id, result); return true;
    } catch (reason: any) { closeError.value = `关闭交互环境失败：${reason.message} 请重试；当前页面将保留。`; return false; }
  })().finally(() => { closing.value = false; closePromise = null; });
  return closePromise;
}
async function leaveEnvironment() {
  leaving.value = true;
  const closed = await closeEnvironment();
  if (!closed) leaving.value = false;
  return closed;
}
async function closeOtherSession(item: any) {
  if (item.id === job.value?.id) return closeEnvironment();
  if (otherClosing.value.includes(item.id)) return;
  otherClosing.value.push(item.id); delete otherCloseErrors.value[item.id];
  try { confirmClosed(item.id, await api(`${prefix}/sessions/${encodeURIComponent(item.id)}/close`, { method: "POST" })); }
  catch (reason: any) { otherCloseErrors.value[item.id] = `关闭失败：${reason.message}`; }
  finally { otherClosing.value = otherClosing.value.filter(id => id !== item.id); }
}
function keepaliveClose(id: string | undefined = job.value?.id): Promise<void> {
  if (!id || confirmedClosed.has(id)) return Promise.resolve();
  const existing = fallbackClosures.get(id); if (existing) return existing;
  const request = (async () => {
    try {
      const response = await fetch(`/api/v1${prefix}/sessions/${encodeURIComponent(id)}/close`, { method: "POST", credentials: "include", keepalive: true });
      if (!response.ok) throw Error(await response.text());
      confirmClosed(id, await response.json());
    } catch (reason: any) { closeError.value = `关闭交互环境未获确认：${reason.message} 请重试关闭。`; }
  })().finally(() => { fallbackClosures.delete(id); });
  fallbackClosures.set(id, request); return request;
}
function keepaliveClosePending(key: string): Promise<void> {
  if (confirmedPendingKeys.has(key)) return Promise.resolve();
  const mapKey = `pending:${key}`, existing = fallbackClosures.get(mapKey); if (existing) return existing;
  const request = (async () => {
    try {
      const response = await fetch(`/api/v1${prefix}/sessions/close-pending`, { method: "POST", credentials: "include", keepalive: true,
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ idempotency_key: key }) });
      if (!response.ok) throw Error(await response.text());
      confirmPendingClosed(key, await response.json());
    } catch (reason: any) { closeError.value = `关闭创建中的交互环境未获确认：${reason.message} 请重试关闭。`; }
  })().finally(() => { fallbackClosures.delete(mapKey); });
  fallbackClosures.set(mapKey, request); return request;
}
function pagehide() {
  suspended.value = true; closeRequested.value = true;
  sessionRevision++;
  clearTimeout(jobTimer); clearTimeout(clockTimer); closeStream();
  if (submissionKey) void keepaliveClosePending(submissionKey);
  void keepaliveClose();
  if (submissionPromise) void submissionPromise.then(result => keepaliveClose(result.id), () => {});
}
async function pageshow(event: PageTransitionEvent) {
  if (!event.persisted || controller.signal.aborted) return;
  suspended.value = false; clearTimeout(clockTimer); tickClock();
  await loadActiveSessions();
  if (job.value?.id) {
    const id = job.value.id, version = sessionRevision;
    try {
      const result = await api(`${prefix}/sessions/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (!controller.signal.aborted && job.value?.id === id && version === sessionRevision) job.value = result;
    }
    catch (reason: any) { if (!controller.signal.aborted) closeError.value = `刷新交互环境状态失败：${reason.message}`; }
  }
}
async function check() {
  if (busy.value || closing.value || leaving.value || suspended.value || frozen.value || !modelId.value || !environmentId.value) return;
  busy.value = true; error.value = ""; preview.value = null;
  const version = revision, inputs = body();
  try {
    const result = await api(`${prefix}/sessions/preview`, { method: "POST", body: JSON.stringify(inputs), signal: controller.signal });
    if (!controller.signal.aborted && revision === version) preview.value = { ...result, body: inputs, key: liveId() };
  } catch (e: any) { if (!controller.signal.aborted) error.value = e.message; }
  finally { busy.value = false; }
}
async function start() {
  if (busy.value || closing.value || leaving.value || suspended.value || frozen.value || !modelId.value || !environmentId.value) return;
  closeRequested.value = false; closeNotice.value = ""; closeError.value = "";
  if (!preview.value) await check();
  if (!preview.value?.plan?.hash || closeRequested.value || closing.value || leaving.value || suspended.value || controller.signal.aborted || busy.value || frozen.value) return;
  busy.value = true; error.value = "";
  const checked = preview.value;
  let creating: Promise<any> | null = null;
  try {
    submissionKey = checked.key;
    creating = api(`${prefix}/sessions`, { method: "POST", headers: { "Idempotency-Key": checked.key }, body: JSON.stringify({ ...checked.body, preview_hash: checked.plan.hash }) });
    submissionPromise = creating;
    const result = await creating;
    if (submissionKey === checked.key) submissionKey = null;
    if (confirmedPendingKeys.has(checked.key)) { if (!confirmedClosed.has(result.id)) void keepaliveClose(result.id); return; }
    if (confirmedClosed.has(result.id) && job.value?.id === result.id) { preview.value = null; return; }
    closeStream(); reconnectAttempt = 0; cameraSeq = 0;
    sessionRevision++;
    state.value = null; pending.value = null; commandUncertain.value = false; transportError.value = "";
    job.value = result; preview.value = null;
    if (suspended.value || controller.signal.aborted) { void keepaliveClose(result.id); return; }
    if (!closeRequested.value && !leaving.value) void pollJob();
  } catch (e: any) { if (!controller.signal.aborted && !confirmedPendingKeys.has(checked.key)) error.value = e.message; }
  finally { if (submissionPromise === creating) { submissionPromise = null; busy.value = false; } }
}
async function pollJob() {
  clearTimeout(jobTimer);
  if (!job.value || jobLoading || controller.signal.aborted || closeRequested.value || suspended.value) return;
  const polledId = job.value.id, polledRevision = sessionRevision;
  jobLoading = true;
  try {
    const result = await api(`${prefix}/sessions/${encodeURIComponent(polledId)}`, { signal: controller.signal });
    if (controller.signal.aborted || closeRequested.value || suspended.value || job.value?.id !== polledId || sessionRevision !== polledRevision) return;
    job.value = result;
    if (result.model_id) modelId.value = result.model_id;
    if (result.environment_id) environmentId.value = result.environment_id;
    if (terminal(result)) closeStream();
    else if (result.run_id && socketRun !== result.run_id && !liveEnded.value) openStream();
  } catch (e: any) { if (!controller.signal.aborted && job.value?.id === polledId && sessionRevision === polledRevision) error.value = e.message; }
  finally {
    jobLoading = false;
    if (!controller.signal.aborted && !closeRequested.value && !suspended.value && !terminal(job.value)) jobTimer = setTimeout(pollJob, job.value?.id === polledId && sessionRevision === polledRevision ? 2000 : 0);
  }
}
function closeStream() {
  clearTimeout(heartbeatTimer); clearTimeout(reconnectTimer); clearTimeout(cameraTimer); clearTimeout(commandTimer);
  cameraTimer = undefined;
  connected.value = false; socketRun = "";
  const previous = socket; socket = null;
  if (previous) {
    previous.onopen = previous.onmessage = previous.onerror = previous.onclose = null;
    try { previous.close(); } catch { /* Local cleanup must still release the frame decoder. */ }
  }
  stage.value?.reset();
}
function openStream() {
  if (controller.signal.aborted || closeRequested.value || suspended.value || !job.value?.run_id || terminal(job.value) || liveEnded.value) return;
  clearTimeout(reconnectTimer);
  const run = job.value.run_id;
  if (socket && socketRun === run) return;
  closeStream(); socketRun = run;
  let ws: WebSocket;
  try { ws = new WebSocket(liveStreamUrl(run)); }
  catch (reason: any) { streamDisconnected(reason.message); return; }
  socket = ws; ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    if (socket !== ws || controller.signal.aborted) return;
    connected.value = true; receivedAt.value = 0;
    transportError.value = "已连接，等待本 Run 的最新状态…";
    sendCamera(); heartbeat();
  };
  ws.onmessage = event => {
    if (socket !== ws || controller.signal.aborted) return;
    try {
      if (event.data instanceof ArrayBuffer) {
        if (!stage.value) throw Error("实时图像画布尚未就绪");
        stage.value.receiveFrame(event.data, seq => { if (socket === ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "frame_ack", seq })); });
        return;
      }
      if (typeof event.data !== "string" || event.data.length > 262144) throw Error("实时消息长度或类型无效");
      const message = JSON.parse(event.data);
      if (message.type === "state") {
        const checked = validateLiveState(message.state, identity(), state.value);
        if (!checked) return;
        state.value = checked; receivedAt.value = performance.now(); now.value = receivedAt.value;
        transportError.value = ""; reconnectAttempt = 0;
        if (pending.value && checked.ack?.id === pending.value.id) {
          pending.value = null; commandSending.value = false; commandUncertain.value = false; error.value = ""; clearTimeout(commandTimer);
        }
        if (liveEnded.value) { clearTimeout(heartbeatTimer); clearTimeout(commandTimer); }
      } else if (message.type === "control_result") {
        if (message.id !== pending.value?.id) return;
        if (typeof message.ok !== "boolean" || !Number.isInteger(message.status)) throw Error("控制回执无效");
        commandSending.value = false;
        if (!message.ok) {
          clearTimeout(commandTimer);
          if (message.status >= 400 && message.status < 500) { pending.value = null; commandUncertain.value = false; error.value = `控制请求被拒绝：${message.detail || message.status}`; }
          else { markUncertain(`控制发送结果未知，可重试同一命令：${message.detail || message.status}`); }
        }
      } else if (message.type === "heartbeat_result" && message.ok === false) {
        throw Error(`会话续租失败：${message.detail || message.status}`);
      } else if (message.type === "error") throw Error(String(message.detail || "实时流服务异常"));
    } catch (reason: any) { frameError(reason.message); }
  };
  ws.onerror = () => { if (socket === ws) transportError.value = "实时连接发生错误，等待重新连接；控制已禁用。"; };
  ws.onclose = () => { if (socket === ws) { socket = null; streamDisconnected(`实时连接已断开，控制已禁用。${transportError.value}`); } };
}
function streamDisconnected(message: string) {
  connected.value = false;
  clearTimeout(heartbeatTimer); clearTimeout(cameraTimer);
  cameraTimer = undefined;
  stage.value?.reset();
  if (pending.value) markUncertain("连接中断，控制结果未知；重新连接后可重试同一命令。");
  if (controller.signal.aborted || closeRequested.value || suspended.value || terminal(job.value) || liveEnded.value) return;
  transportError.value = message;
  if (reconnectAttempt >= 8) { transportError.value += " 自动重连未成功，请使用重连按钮。"; return; }
  const delay = Math.min(500 * 2 ** reconnectAttempt++, 5000);
  reconnectTimer = setTimeout(() => { socketRun = ""; openStream(); }, delay);
}
function reconnect() { reconnectAttempt = 0; closeStream(); openStream(); }
function frameError(message: string) {
  transportError.value = message; connected.value = false;
  socket?.close(1003, "Invalid live stream");
}
function heartbeat() {
  clearTimeout(heartbeatTimer);
  if (!connected.value || socket?.readyState !== WebSocket.OPEN || controller.signal.aborted || closeRequested.value || suspended.value || terminal(job.value) || liveEnded.value) return;
  socket.send(JSON.stringify({ type: "heartbeat" }));
  heartbeatTimer = setTimeout(heartbeat, 3000);
}
function cameraChanged(value: LiveCamera) {
  camera = value;
  if (closeRequested.value || suspended.value) return;
  if (!cameraTimer) cameraTimer = setTimeout(() => { cameraTimer = undefined; sendCamera(); }, 34);
}
function sendCamera() {
  if (!closeRequested.value && !suspended.value && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "camera", seq: ++cameraSeq, ...camera }));
}
function command(action: string, task?: string) {
  if (!canControl.value || (state.value?.status === "fallen" && !["reset", "stop"].includes(action))) return;
  const requested = { id: liveId(), action, ...(task ? { task } : {}) };
  pending.value = requested; sendCommand(requested);
}
function retryCommand() {
  if (!pending.value || commandSending.value || !commandUncertain.value || stale.value || terminal(job.value) || liveEnded.value) return;
  sendCommand(pending.value);
}
function markUncertain(message: string) { clearTimeout(commandTimer); commandSending.value = false; commandUncertain.value = true; error.value = message; }
function sendCommand(requested: { id: string; action: string; task?: string }) {
  commandSending.value = true; commandUncertain.value = false; error.value = "";
  try {
    if (!connected.value || socket?.readyState !== WebSocket.OPEN) throw Error("实时连接不可用");
    socket.send(JSON.stringify({ type: "control", command: requested }));
    clearTimeout(commandTimer);
    commandTimer = setTimeout(() => {
      if (pending.value?.id === requested.id) markUncertain("尚未收到实际应用确认，可重试同一控制命令。");
    }, 5000);
  } catch (reason: any) { if (pending.value?.id === requested.id) markUncertain(`控制发送结果未知，可重试同一命令：${reason.message}`); }
}
function number(value: number | undefined, digits = 2) { return value == null ? "—" : value.toFixed(digits); }
function taskLabel(task: string) { return capabilities.value?.tasks?.find((item: any) => item.id === task)?.label || task || "等待状态"; }
</script>

<template>
  <div>
    <div class="page-head"><h2>实时仿真验证</h2><router-link to="/deploy/bundles">仿真交付包</router-link></div>
    <p class="muted">先启动仿真环境，再在右侧实时切换站立与前行任务。切换保留当前姿态和历史观测，无需重新启动仿真。此处仅记录交互诊断。</p>
    <div v-if="job || busy" class="environment-close">
      <button type="button" class="danger" :disabled="closing" @click="closeEnvironment">{{ closing ? '正在关闭交互环境…' : '关闭交互环境' }}</button>
      <span class="muted xs">立即释放计算资源；离开此页面也会自动关闭。已落盘的诊断记录会保留。</span>
    </div>
    <p v-if="closing" role="status">{{ busy ? '正在取消创建请求并等待计算资源释放确认…' : '正在停止环境并等待计算资源释放确认…' }}</p>
    <p v-if="closeNotice" role="status">{{ closeNotice }}</p>
    <p v-if="closeError" class="error" role="alert">{{ closeError }}</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="!capabilities && !error" role="status">读取兼容模型与环境…</p>
    <p v-if="capabilities && !capabilities.models?.length">尚无兼容的已导出模型。请先在模型详情完成原生导出。</p>
    <p v-if="capabilities?.image_ready === false" class="error" role="alert">实时仿真任务镜像尚未就绪。</p>
    <section class="panel selection" aria-label="冻结仿真输入">
      <label>模型<select v-model="modelId" :disabled="frozen || busy"><option value="" disabled>选择已导出模型</option><option v-for="item in capabilities?.models || []" :key="item.id" :value="item.id">{{ item.name || item.id }}</option></select></label>
      <label>仿真环境<select v-model="environmentId" :disabled="frozen || busy"><option value="" disabled>选择兼容环境</option><option v-for="item in environments" :key="item.id" :value="item.id">{{ item.name }} · {{ item.backend }}</option></select></label>
      <label>时限（秒）<input v-model.number="duration" type="number" min="30" max="600" :disabled="frozen || busy" /></label>
      <label>随机种子<input v-model.number="seed" type="number" min="0" step="1" :disabled="frozen || busy" /></label>
      <button type="button" class="primary" :disabled="busy || closing || leaving || frozen || !modelId || !environmentId || capabilities?.image_ready === false" @click="start">{{ busy ? '正在处理…' : '启动仿真环境' }}</button>
      <button type="button" :disabled="busy || closing || leaving || frozen || !modelId || !environmentId || capabilities?.image_ready === false" @click="check">检查仿真配置</button>
      <div v-if="preview" class="preview"><details><summary>仿真配置已就绪 · {{ preview.plan.hash }}</summary><pre>{{ JSON.stringify({ plan: preview.plan, resources: preview.resources }, null, 2) }}</pre></details></div>
    </section>
    <section v-if="otherSessions.length || activeSessionsError" class="panel other-sessions" aria-label="其他活动交互环境">
      <h3>其他活动交互环境</h3>
      <p v-if="activeSessionsError" class="error" role="alert">{{ activeSessionsError }}</p>
      <div v-for="item in otherSessions" :key="item.id" class="active-session">
        <span class="mono xs">{{ item.id }} · {{ statusLabels[item.status] || item.status }} · {{ item.model_id }} · {{ item.backend || item.environment_id }}</span>
        <router-link :to="{ path: '/deploy/simulation', query: { session_job: item.id } }">重连此环境</router-link>
        <button type="button" :disabled="otherClosing.includes(item.id)" @click="closeOtherSession(item)">{{ otherClosing.includes(item.id) ? '正在关闭…' : '关闭交互环境' }}</button>
        <p v-if="otherCloseErrors[item.id]" class="error" role="alert">{{ otherCloseErrors[item.id] }}</p>
      </div>
    </section>
    <p v-if="job" class="mono xs">Job {{ job.id }} · {{ statusLabels[job.status] || job.status }} <router-link v-if="job.run_id" :to="`/train/runs/${job.run_id}`">Run {{ job.run_id }}</router-link> · <router-link :to="{ path: '/deploy/simulation', query: { session_job: job.id } }">重连此会话</router-link></p>
    <p v-if="job?.error || state?.error" class="error" role="alert">{{ state?.error || job?.error }}</p>
    <div class="live-layout">
      <section class="panel stage-panel" aria-label="实时机器人姿态">
        <div class="page-head"><strong>{{ closeNotice ? '交互环境已关闭' : job ? statusLabels[state?.status || 'initializing'] : '等待启动仿真环境' }}</strong><span v-if="state" class="mono xs">episode {{ state.episode }} · step {{ state.step }}</span></div>
        <p v-if="job && stale && !liveEnded && !terminal(job)" class="error" role="status">{{ state ? '实时状态已断流，控制已禁用。' : '等待仿真初始化与第一帧真实状态…' }}</p>
        <p v-if="transportError" class="error" role="alert">{{ transportError }}</p>
        <button v-if="job?.run_id && !connected && !closeRequested && !liveEnded && !terminal(job)" type="button" @click="reconnect">重新连接实时画面</button>
        <NativeSimulationStage ref="stage" @camera="cameraChanged" @error="frameError" />
        <p class="muted xs">画面由原生仿真渲染器实时输出。帧龄按浏览器接收后计时，不代表跨机器的端到端时延。</p>
      </section>
      <aside class="panel live-controls" aria-label="仿真控制">
        <h3>仿真内切换任务</h3><p v-if="!job" class="muted xs">环境启动后，在此直接切换任务。环境默认从站立开始。</p><p>实际任务：<strong>{{ taskLabel(activeTask) }}</strong></p>
        <p v-if="pending" role="status">请求：{{ pending.action === 'task' ? taskLabel(pending.task || '') : pending.action }} · 等待实际应用确认</p>
        <button v-if="pending && commandUncertain" type="button" :disabled="commandSending || stale || liveEnded || terminal(job)" @click="retryCommand">重试同一控制命令</button>
        <div class="task-buttons"><button v-for="task in capabilities?.tasks || []" :key="task.id" type="button" :aria-pressed="activeTask === task.id" :disabled="!canControl || state?.status === 'fallen'" @click="command('task', task.id)">{{ task.label }}</button></div>
        <p class="muted xs">切换任务保留当前姿态、上一动作与历史；重置按钮开始同一种子的下一轮。</p>
        <div class="control-buttons"><button type="button" :disabled="!canControl || !['running', 'paused'].includes(state?.status || '')" @click="command(state?.status === 'paused' ? 'resume' : 'pause')">{{ state?.status === 'paused' ? '继续' : '暂停' }}</button><button type="button" :disabled="!canControl" @click="command('reset')">重置</button><button type="button" :disabled="!canControl" @click="command('stop')">结束并保存录像</button></div>
        <p v-if="state?.status === 'fallen'" class="error" role="alert">机器人跌倒，物理已暂停。可重置后继续或停止会话。</p>
        <dl><dt>目标 / 实际前向速度</dt><dd>{{ number(targetSpeed) }} / {{ number(state?.velocity[0]) }} m/s</dd><dt>侧向速度</dt><dd>{{ number(state?.velocity[1]) }} m/s</dd><dt>推理时延</dt><dd>{{ number(state?.inference_ms) }} ms</dd><dt>实时率</dt><dd>{{ number(state?.realtime_factor) }} ×</dd><dt>高度 / 倾角</dt><dd>{{ number(state?.height) }} m / {{ number(state?.tilt) }} rad</dd><dt>仿真 / 实际用时</dt><dd>{{ number(state?.sim_seconds, 1) }} / {{ number(state?.wall_seconds, 1) }} s</dd><dt>位置 / 速度 / 力矩越限次数</dt><dd>{{ state?.limits.joint ?? '—' }} / {{ state?.limits.velocity ?? '—' }} / {{ state?.limits.torque ?? '—' }}</dd></dl>
        <p v-if="state && state.realtime_factor < .9" class="muted">当前仿真慢于实时，任务按实际物理步生效。</p>
        <p v-if="state && Object.values(state.limits).some(value => value > 0)" class="error">检测到越限；次数为本会话累计值。</p>
        <p v-if="state?.ack" class="mono xs">已应用 {{ state.ack.action }} · step {{ state.ack.applied_step }} · episode {{ state.ack.episode }}</p>
      </aside>
    </div>
    <section v-if="liveEnded || terminal(job)" class="panel recordings" aria-label="会话证据">
      <h3>诊断证据</h3><p v-if="!terminal(job)" role="status">实时会话已结束，正在整理轨迹、图表和录像…</p><p>Job 执行成功仅表示诊断产物已收集，机器人任务表现以实际记录为准。</p>
      <p v-if="terminal(job) && !job?.files?.length" class="error">尚无完整可访问产物；不能据此宣称证据完整。</p>
      <div v-for="file in job?.files || []" :key="file.path"><a :href="file.url" target="_blank" rel="noopener">{{ file.path }}</a><video v-if="/\.mp4$/i.test(file.path)" :src="file.url" controls preload="metadata" :aria-label="file.path" @error="mediaErrors[file.path] = true" /><img v-if="/\.(png|svg|jpe?g)$/i.test(file.path)" :src="file.url" :alt="file.path" loading="lazy" @error="mediaErrors[file.path] = true" /><p v-if="mediaErrors[file.path]" class="error" role="alert">媒体加载失败：{{ file.path }}</p></div>
    </section>
  </div>
</template>

<style scoped>
.selection { display: flex; flex-wrap: wrap; align-items: end; gap: .8rem; margin-bottom: 1rem; }
.environment-close, .active-session { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; margin-bottom: 1rem; }
.environment-close .danger { border-color: #e47777; color: #efaaaa; }
.other-sessions { margin-bottom: 1rem; }
.selection label { display: grid; gap: .35rem; flex: 1 1 160px; }
.selection input, .selection select { width: 100%; }
.preview { width: 100%; }
.preview pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.live-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 1rem; }
.stage-panel { min-width: 0; }
.task-buttons, .control-buttons { display: flex; flex-wrap: wrap; gap: .5rem; }
[aria-pressed="true"] { border-color: var(--accent, #59bfff); }
dl { display: grid; grid-template-columns: 1fr; gap: .25rem; } dt { color: var(--muted); margin-top: .4rem; } dd { margin: 0; font-variant-numeric: tabular-nums; }
.recordings { margin-top: 1rem; } .recordings video, .recordings img { display: block; width: 100%; max-width: 900px; margin: .6rem 0; }
@media (max-width: 1000px) { .live-layout { grid-template-columns: 1fr; } }
</style>
