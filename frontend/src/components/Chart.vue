<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
const props = defineProps<{
  points: { step: number; value: number }[];
  title?: string;
  playhead?: number;
}>();
const el = ref<HTMLDivElement | null>(null);
let chart: any = null;
async function draw() {
  if (!el.value) return;
  const echarts = await import("echarts");
  if (!el.value) return;
  chart ||= echarts.init(el.value);
  const markLine =
    props.playhead != null
      ? { symbol: "none", lineStyle: { color: "#ffc857" }, data: [{ xAxis: props.playhead }] }
      : undefined;
  chart.setOption({
    backgroundColor: "transparent",
    animation: props.playhead == null,
    title: { text: props.title || "", textStyle: { color: "#9fb4c2", fontSize: 12 } },
    xAxis: { type: "value", axisLabel: { color: "#9fb4c2" } },
    yAxis: { type: "value", axisLabel: { color: "#9fb4c2" } },
    series: [
      {
        type: "line",
        showSymbol: props.points?.length === 1,
        data: (props.points || []).map((p) => [p.step, p.value]),
        lineStyle: { color: "#53d4ff" },
        markLine,
      },
    ],
    grid: { left: 40, right: 12, top: 28, bottom: 24 },
  });
}
onMounted(draw);
onUnmounted(() => { chart?.dispose(); chart = null; });
watch(() => [props.points, props.playhead], draw, { deep: true });
</script>
<template>
  <div ref="el" style="width:100%;height:220px" />
</template>
