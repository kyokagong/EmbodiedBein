<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/evals/suites")).items || []; });
</script>
<template>
  <div>
    <h2>评估套件</h2>
    <p class="muted">LIBERO 评估使用固定仿真资产与初始化状态；LeRobot 示教 Dataset 不等于闭环测试套件。</p>
    <EbTable :columns="[{key:'id',label:'id'},{key:'name',label:'name'},{key:'embodiment_env_id',label:'env'},{key:'frozen',label:'frozen'}]" :rows="rows" />
    <div v-for="r in rows.filter(r => r.spec?.benchmark_assets)" :key="r.id" class="muted xs">
      {{ r.name }} · 资产 {{ r.spec.benchmark_assets.repo }} @ {{ r.spec.benchmark_assets.revision }}
    </div>
  </div>
</template>
