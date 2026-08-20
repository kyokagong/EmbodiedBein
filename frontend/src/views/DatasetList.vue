<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => {
  rows.value = (await api("/datasets")).items || [];
});
</script>
<template>
  <div>
    <h2>数据集</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'name',label:'name'},{key:'format',label:'format'},{key:'license',label:'license'}]" :rows="rows">
      <template #id="{ row }"><router-link :to="`/data/${row.id}`">{{ row.id }}</router-link></template>
    </EbTable>
  </div>
</template>
