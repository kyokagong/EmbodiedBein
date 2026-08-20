<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const versions = ref<any[]>([]);
onMounted(async () => {
  versions.value = (await api("/models")).versions || [];
});
</script>
<template>
  <div>
    <h2>模型</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'name',label:'name'},{key:'contract_id',label:'contract'},{key:'provenance',label:'provenance'}]" :rows="versions">
      <template #id="{ row }"><router-link :to="`/models/${row.id}`">{{ row.id }}</router-link></template>
      <template #name="{ row }"><router-link :to="`/models/${row.id}`">{{ row.name }}</router-link></template>
    </EbTable>
  </div>
</template>
