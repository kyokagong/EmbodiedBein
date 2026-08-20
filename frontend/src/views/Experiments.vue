<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/experiments")).items || []; });
</script>
<template>
  <div>
    <h2>实验</h2>
    <EbTable :columns="[{key:'id',label:'id'},{key:'name',label:'name'},{key:'contract_id',label:'contract'}]" :rows="rows" />
  </div>
</template>
