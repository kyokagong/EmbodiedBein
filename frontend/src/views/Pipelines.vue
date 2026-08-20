<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
import EbTable from "../components/EbTable.vue";
const rows = ref<any[]>([]);
onMounted(async () => { rows.value = (await api("/pipelines")).items || []; });
</script>
<template>
  <div>
    <h2>流水线</h2>
    <p class="muted">steps[] 顺序提交，下一步等 SUCCEEDED。</p>
    <EbTable :columns="[{key:'id',label:'id'},{key:'name',label:'name'}]" :rows="rows" />
  </div>
</template>
