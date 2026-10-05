<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { api } from "../api/client";
const reports = ref<any[]>([]); const left = ref(""); const right = ref("");
const leftCases = ref<any[]>([]); const rightCases = ref<any[]>([]);
let requestVersion = 0;
const a = computed(() => reports.value.find(r => r.id === left.value));
const b = computed(() => reports.value.find(r => r.id === right.value));
const compatible = computed(() => !!a.value && !!b.value && a.value.id !== b.value.id &&
  a.value.suite_id === b.value.suite_id && a.value.metrics?.profile === b.value.metrics?.profile &&
  !!a.value.gate_results?.protocol && JSON.stringify(a.value.gate_results.protocol) === JSON.stringify(b.value.gate_results?.protocol) &&
  a.value.gate_results?.validity === 'complete' && b.value.gate_results?.validity === 'complete' &&
  leftCases.value.length === rightCases.value.length && leftCases.value.length > 0 &&
  leftCases.value.every(ep => rightCases.value.some(other => other.case_id === ep.case_id && other.reset_state_id === ep.reset_state_id)));
const paired = computed(() => {
  if (!compatible.value) return null;
  const rhs = new Map(rightCases.value.map(ep => [ep.case_id, ep]));
  const counts = { aOnly: 0, bOnly: 0, both: 0, neither: 0 };
  for (const ep of leftCases.value) {
    const other: any = rhs.get(ep.case_id);
    if (ep.success && other.success) counts.both++;
    else if (ep.success) counts.aOnly++;
    else if (other.success) counts.bOnly++;
    else counts.neither++;
  }
  return counts;
});
onMounted(async () => {
  reports.value = (await api('/evals/reports')).items || [];
  left.value = reports.value[0]?.id || ''; right.value = reports.value[1]?.id || '';
});
watch([left, right], async () => {
  const version = ++requestVersion;
  leftCases.value = rightCases.value = [];
  if (!left.value || !right.value) return;
  try {
    const [x, y] = await Promise.all([api(`/evals/reports/${left.value}/episodes`), api(`/evals/reports/${right.value}/episodes`)]);
    if (version === requestVersion) { leftCases.value = x.items || []; rightCases.value = y.items || []; }
  } catch { /* Keep an explicit unavailable state. */ }
});
</script>
<template>
  <div><h2>评估报告对比</h2><p class="muted">先核对套件、规模和实际案例，再查看成对变化；这里只做描述性比较，不从区间重叠推断显著性。</p>
    <div class="row"><label>报告 A <select v-model="left"><option v-for="r in reports" :key="r.id" :value="r.id">{{ r.id }} · {{ r.suite_id }}</option></select></label>
      <label>报告 B <select v-model="right"><option v-for="r in reports" :key="r.id" :value="r.id">{{ r.id }} · {{ r.suite_id }}</option></select></label></div>
    <div v-if="a && b" class="panel"><p v-if="compatible">套件、执行协议与已记录案例一致；此处仅报告同案例的描述性变化。</p>
      <p v-else class="muted">案例、镜像或推理协议证据不一致、缺失，当前仅允许并列阅读。</p>
      <table><thead><tr><th>维度</th><th>A</th><th>B</th></tr></thead><tbody>
        <tr><td>套件</td><td>{{ a.suite_id }}</td><td>{{ b.suite_id }}</td></tr>
        <tr><td>结果</td><td>{{ a.wilson_ci?.label || '无可信样本量' }}</td><td>{{ b.wilson_ci?.label || '无可信样本量' }}</td></tr>
        <tr><td>证据</td><td>{{ a.gate_results?.validity || '历史未验证' }}</td><td>{{ b.gate_results?.validity || '历史未验证' }}</td></tr>
      </tbody></table>
      <p v-if="paired">同成 {{ paired.both }} · A 独成 {{ paired.aOnly }} · B 独成 {{ paired.bOnly }} · 同败 {{ paired.neither }}</p>
    </div>
  </div>
</template>
