<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { createLiveFrameSink, defaultLiveCamera, type LiveCamera, type LiveFrameMetadata } from "../utils/ufoLive";

const emit = defineEmits<{ camera: [camera: LiveCamera]; error: [message: string] }>();
const canvas = ref<HTMLCanvasElement | null>(null);
const camera = ref(defaultLiveCamera());
const metadata = ref<LiveFrameMetadata | null>(null);
const drawnAt = ref(0);
const error = ref("");
const age = ref(0), fps = ref(0);
let receivedAt = 0, arrivals: number[] = [], dragging: { id: number; x: number; y: number } | null = null;
let timer: ReturnType<typeof setTimeout> | undefined;
let mounted = false;
const sink = createLiveFrameSink({
  decode: frame => createImageBitmap(new Blob([frame.jpeg as BlobPart], { type: "image/jpeg" })),
  draw(bitmap, info, received) {
    const context = canvas.value?.getContext("2d");
    if (!context) throw Error("浏览器无法创建实时图像画布");
    context.drawImage(bitmap, 0, 0, info.width, info.height);
    drawnAt.value = performance.now();
    metadata.value = info; receivedAt = received; age.value = performance.now() - received;
  },
  error(reason) { error.value = `原生图像显示失败：${reason.message}`; emit("error", error.value); },
});
function tick() {
  const now = performance.now();
  arrivals = arrivals.filter(time => now - time < 1000);
  fps.value = arrivals.length > 1 ? (arrivals.length - 1) * 1000 / (arrivals.at(-1)! - arrivals[0]) : 0;
  if (metadata.value) age.value = Math.max(0, now - receivedAt);
  if (mounted) timer = setTimeout(tick, 250);
}
onMounted(() => { mounted = true; tick(); });
onUnmounted(() => { mounted = false; clearTimeout(timer); sink.dispose(); });
function receiveFrame(data: ArrayBuffer, acknowledge: (seq: number) => void) {
  arrivals.push(performance.now());
  if (arrivals.length > 120) arrivals.shift();
  sink.push(data, acknowledge);
}
function reset() {
  sink.reset(); metadata.value = null; error.value = ""; arrivals = []; fps.value = 0;
  canvas.value?.getContext("2d")?.clearRect(0, 0, 640, 480);
}
defineExpose({ receiveFrame, reset });
function updateCamera(change: Partial<LiveCamera>) {
  camera.value = { azimuth: Math.max(-360, Math.min(360, change.azimuth ?? camera.value.azimuth)),
    elevation: Math.max(-85, Math.min(-5, change.elevation ?? camera.value.elevation)),
    distance: Math.max(1, Math.min(8, change.distance ?? camera.value.distance)) };
  emit("camera", { ...camera.value });
}
function pointerDown(event: PointerEvent) {
  if (event.button !== 0) return;
  (event.currentTarget as HTMLCanvasElement).setPointerCapture(event.pointerId);
  dragging = { id: event.pointerId, x: event.clientX, y: event.clientY };
}
function pointerMove(event: PointerEvent) {
  if (!dragging || dragging.id !== event.pointerId) return;
  updateCamera({ azimuth: camera.value.azimuth - (event.clientX - dragging.x) * .4, elevation: camera.value.elevation - (event.clientY - dragging.y) * .25 });
  dragging.x = event.clientX; dragging.y = event.clientY;
}
function pointerUp(event: PointerEvent) {
  if (dragging?.id !== event.pointerId) return;
  dragging = null;
  const target = event.currentTarget as HTMLCanvasElement;
  if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
}
function wheel(event: WheelEvent) { updateCamera({ distance: camera.value.distance * Math.exp(Math.max(-100, Math.min(100, event.deltaY)) * .002) }); }
function keydown(event: KeyboardEvent) {
  const change: Record<string, Partial<LiveCamera>> = {
    ArrowLeft: { azimuth: camera.value.azimuth + 5 }, ArrowRight: { azimuth: camera.value.azimuth - 5 },
    ArrowUp: { elevation: camera.value.elevation + 5 }, ArrowDown: { elevation: camera.value.elevation - 5 },
    "+": { distance: camera.value.distance - .2 }, "=": { distance: camera.value.distance - .2 }, "-": { distance: camera.value.distance + .2 },
    Home: defaultLiveCamera(),
  };
  if (change[event.key]) { event.preventDefault(); updateCamera(change[event.key]); }
}
</script>

<template>
  <div class="native-stage">
    <canvas ref="canvas" width="640" height="480" tabindex="0" role="img" aria-label="原生仿真实时画面；方向键旋转，加减键缩放，Home 重置相机"
      :data-frame-seq="metadata?.seq" :data-frame-step="metadata?.step" :data-frame-episode="metadata?.episode" :data-camera-seq="metadata?.camera_seq"
      :data-source-updated-at="metadata?.updated_at" :data-rendered-at="metadata?.rendered_at" :data-drawn-at="metadata ? drawnAt : undefined"
      @pointerdown="pointerDown" @pointermove="pointerMove" @pointerup="pointerUp" @pointercancel="pointerUp" @lostpointercapture="dragging = null" @wheel.prevent="wheel" @keydown="keydown" />
    <p v-if="error" class="overlay error" role="alert">{{ error }}</p>
    <p v-else-if="!metadata" class="overlay" role="status">等待原生渲染的实时图像…</p>
  </div>
  <div class="camera-bar"><button type="button" @click="updateCamera(defaultLiveCamera())">重置相机</button><span class="muted xs">拖动旋转，滚轮缩放；方向键、加减键亦可操作。相机跟随机器人。</span></div>
  <p v-if="metadata" class="mono xs">接收 {{ fps.toFixed(1) }} FPS · 本帧接收后 {{ (age / 1000).toFixed(2) }} s · 画面 step {{ metadata.step }} · 相机 {{ metadata.camera_seq }} · 渲染 {{ metadata.render_ms.toFixed(1) }} / 编码 {{ metadata.encode_ms.toFixed(1) }} ms</p>
</template>

<style scoped>
.native-stage { position: relative; background: #0d1720; border-radius: .5rem; overflow: hidden; }
canvas { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; max-height: 65vh; touch-action: none; cursor: grab; }
canvas:active { cursor: grabbing; } canvas:focus-visible { outline: 2px solid var(--accent, #59bfff); outline-offset: -2px; }
.overlay { position: absolute; left: 1rem; right: 1rem; top: .5rem; background: #152330ee; padding: .75rem; overflow-wrap: anywhere; pointer-events: none; }
.camera-bar { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; margin-top: .75rem; }
</style>
