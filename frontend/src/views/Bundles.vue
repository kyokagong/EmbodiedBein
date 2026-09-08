<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";

const rows = ref<any[]>([]);
const selectedBundle = ref<any>(null);
const promoteStage = ref("S0");
const confirmId = ref("");
const err = ref("");
const msg = ref("");
const showPromoteModal = ref(false);

onMounted(load);

async function load() {
  rows.value = (await api("/bundles")).items || [];
}

async function sign(id: string) {
  err.value = "";
  msg.value = "";
  try {
    await api(`/bundles/${id}/sign`, { method: "POST", headers: { "X-Confirm": id } });
    msg.value = `策略包 ${id} 签名成功（已通过 G3/G4 与数值自检门禁）`;
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}

function openPromote(b: any) {
  selectedBundle.value = b;
  promoteStage.value = b.stage === "S0" ? "S1" : b.stage === "S1" ? "S2" : "S0";
  confirmId.value = "";
  err.value = "";
  msg.value = "";
  showPromoteModal.value = true;
}

async function executePromote() {
  if (confirmId.value !== selectedBundle.value?.id) {
    err.value = "请输入完整的策略包 ID 以确认物理放行";
    return;
  }
  err.value = "";
  try {
    await api(`/releases/${selectedBundle.value.id}/promote`, {
      method: "POST",
      body: JSON.stringify({ stage: promoteStage.value }),
      headers: { "X-Confirm-Promote": confirmId.value },
    });
    msg.value = `策略包已成功放行至 ${promoteStage.value}`;
    showPromoteModal.value = false;
    await load();
  } catch (e: any) {
    err.value = e.message;
  }
}
</script>

<template>
  <div class="bundles-page">
    <div class="page-head">
      <div>
        <p class="crumb">部署与真机 / 策略包</p>
        <h2>策略包交付与真机放行 (PolicyBundles .ebp)</h2>
      </div>
      <router-link to="/deploy/robots/robot_bhl_01" class="btn-link">🤖 机器人台账与遥测</router-link>
    </div>
    <p class="muted small">策略包是训练与真机之间的唯一交付物，自包含 manifest.yaml、policy.onnx、门禁报告与 HMAC 签名。平台严禁直接向真机下发未签名的原始权重。</p>
    <p class="error" v-if="err">{{ err }}</p>
    <p class="ok-msg" v-if="msg">{{ msg }}</p>

    <!-- 策略包列表 -->
    <div class="panel" style="padding:0; overflow:hidden">
      <table>
        <thead>
          <tr>
            <th>策略包 ID</th>
            <th>本体环境</th>
            <th>放行阶段 (Stage)</th>
            <th>签名状态</th>
            <th>安全约束 (Safety Limits)</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="b in rows" :key="b.id">
            <td>
              <div class="mono font-bold">{{ b.id }}</div>
              <div class="muted xs">CAS: {{ (b.manifest?.provenance?.commit || '').slice(0, 8) || 'verified' }}</div>
            </td>
            <td><span class="mono small">{{ b.embodiment_env_id }}</span></td>
            <td>
              <span class="badge" :class="b.stage === 'S2' ? 'b-green' : b.stage === 'S1' ? 'b-blue' : 'b-amber'">
                {{ b.stage || 'S0' }} {{ b.stage === 'S2' ? '全功率' : b.stage === 'S1' ? '限速' : '系留空载' }}
              </span>
            </td>
            <td>
              <span class="badge" :class="b.signed ? 'b-green' : 'b-amber'">
                {{ b.signed ? '✓ 已签名认证' : '未签名 (草稿)' }}
              </span>
            </td>
            <td class="small">
              <code>clip={{ b.manifest?.safety?.action_clip ?? '0.375' }}</code> · 
              <code>watchdog={{ b.manifest?.safety?.watchdog_ms ?? '100' }}ms</code>
            </td>
            <td>
              <div class="row" style="gap:6px">
                <button type="button" class="btn-sm" v-if="!b.signed" @click="sign(b.id)">✍️ 签名确认</button>
                <button type="button" class="btn-sm b-cyan-btn" @click="openPromote(b)">🚀 阶段放行</button>
              </div>
            </td>
          </tr>
          <tr v-if="!rows.length">
            <td colspan="6" class="text-center text-muted" style="padding:2rem">暂无已打包的策略包</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 分级放行确认弹窗 -->
    <div class="modal-overlay" v-if="showPromoteModal">
      <div class="modal-card">
        <h3>物理确认分级放行 (Staged Promotion)</h3>
        <p class="muted small">
          目标策略包: <code class="mono">{{ selectedBundle?.id }}</code> @ <code>{{ selectedBundle?.embodiment_env_id }}</code>
        </p>

        <div class="safety-warning">
          ⚠️ <b>安全红线</b>：放行前请确保急停手柄 (E-Stop) 通畅，周围 2 米无障碍物。
        </div>

        <div class="form-grid" style="margin:1rem 0">
          <span>目标放行阶段</span>
          <select v-model="promoteStage">
            <option value="S0">S0：系留空载（必须挂吊带，禁止落地）</option>
            <option value="S1">S1：平地限速（速度 ≤ 0.3 m/s，人工手持急停）</option>
            <option value="S2">S2：全功率放行（自主作业）</option>
          </select>

          <span>输入策略包 ID 确认</span>
          <input v-model="confirmId" :placeholder="selectedBundle?.id" />
        </div>

        <p class="error" v-if="err">{{ err }}</p>

        <div class="row" style="justify-content:flex-end; gap:8px">
          <button type="button" @click="showPromoteModal = false">取消</button>
          <button type="button" class="primary" @click="executePromote">确认放行并记录审计</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bundles-page { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.font-bold { font-weight: 600; }
.text-muted { color: var(--muted, #9fb4c2); }
.ok-msg { background: rgba(83, 229, 163, 0.15); border: 1px solid #53e5a3; color: #53e5a3; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; margin-bottom: 1rem; }
.btn-sm { padding: 3px 8px; font-size: 0.8rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 4px; cursor: pointer; }
.b-cyan-btn { background: rgba(91, 140, 255, 0.15); border-color: #5b8cff; color: #5b8cff; }
.btn-link { padding: 4px 12px; font-size: 0.85rem; background: rgba(83,212,255,0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 6px; text-decoration: none; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-family: var(--mono, monospace); }
.b-green { background: rgba(83, 229, 163, 0.15); color: #53e5a3; }
.b-blue { background: rgba(91, 140, 255, 0.15); color: #5b8cff; }
.b-amber { background: rgba(255, 200, 87, 0.15); color: #ffc857; }
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; }
.modal-card { background: var(--panel, #0d1924); border: 1px solid var(--line, #284052); border-radius: 12px; padding: 1.5rem; max-width: 540px; width: 90%; }
.safety-warning { background: rgba(255, 200, 87, 0.15); border: 1px solid #ffc857; color: #ffc857; padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; margin-top: 0.8rem; }
.primary { background: var(--cyan, #53d4ff); color: #071018; font-weight: 700; border: none; }
</style>
