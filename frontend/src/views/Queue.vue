<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api } from "../api/client";
const q = ref<any>(null);
const gpus = ref<any>(null);
onMounted(async () => {
  q.value = await api("/resources/queue");
  gpus.value = await api("/resources/gpus");
});
</script>
<template>
  <div>
    <h2>队列与资源</h2>
    <pre>{{ JSON.stringify(gpus, null, 2) }}</pre>
    <table>
      <thead><tr><th>job</th><th>kind</th><th>status</th><th>priority</th></tr></thead>
      <tbody>
        <tr v-for="j in q?.items || []" :key="j.id">
          <td>{{ j.id }}</td><td>{{ j.kind }}</td><td>{{ j.status }}</td><td>{{ j.priority }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
