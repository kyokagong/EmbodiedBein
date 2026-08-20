<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import EmptyState from "../components/EmptyState.vue";
const route = useRoute();
const data = ref<any>(null);
onMounted(async () => {
  data.value = await api(`/models/${route.params.id}`);
});
</script>
<template>
  <div>
    <h2>{{ data?.name || route.params.id }}</h2>
    <p class="muted">{{ data?.contract_id || "待人工归属" }} · {{ data?.provenance }}</p>
    <EmptyState v-if="!data" title="加载中" />
    <div class="kpis" v-else>
      <div class="card">
        <div class="l">I/O</div>
        <div class="v">{{ data.io_signature?.obs ?? "?" }} / {{ data.io_signature?.act ?? "?" }}</div>
      </div>
      <div class="card">
        <div class="l">契约</div>
        <div class="v">{{ data.contract?.obs_dim ?? "—" }} / {{ data.contract?.action_dim ?? "—" }}</div>
      </div>
      <div class="card">
        <div class="l">G3</div>
        <div class="v">{{ data.contract_check ? (data.contract_check.passed ? "通过" : "失败") : "待归属" }}</div>
      </div>
      <div class="card" v-if="data.card">
        <div class="l">来源</div>
        <div class="v">{{ data.card.source_path || data.card.note || "—" }}</div>
      </div>
    </div>
  </div>
</template>
