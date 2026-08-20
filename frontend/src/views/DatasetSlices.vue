<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const route = useRoute();
const rows = ref<any[]>([]);
onMounted(async () => {
  const ds = await api(`/datasets/${route.params.id}`);
  rows.value = ds.versions || [];
});
</script>
<template>
  <div>
    <h2>切片</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'version',label:'version'},{key:'content_hash',label:'hash'}]" :rows="rows" />
  </div>
</template>
