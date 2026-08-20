<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
const err = ref("");
onMounted(async () => { rows.value = (await api("/bundles")).items || []; });
async function sign(id: string) {
  try {
    await api(`/bundles/${id}/sign`, { method: "POST", headers: { "X-Confirm": id } });
    rows.value = (await api("/bundles")).items || [];
  } catch (e: any) { err.value = e.message; }
}
</script>
<template>
  <div>
    <h2>策略包</h2>
    <p class="error">{{ err }}</p>
    <EbTable :columns="[{key:'id',label:'id'},{key:'contract_id',label:'contract'},{key:'stage',label:'stage'},{key:'signed',label:'signed'}]" :rows="rows">
      <template #id="{ row }">
        <span>{{ row.id }}</span>
        <button v-if="!row.signed" @click="sign(row.id)">签名确认</button>
      </template>
    </EbTable>
  </div>
</template>
