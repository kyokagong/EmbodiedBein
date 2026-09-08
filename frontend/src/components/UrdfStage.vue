<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from "vue";

const props = defineProps<{
  embodimentId: string;
  preview: any;
  envs?: any[];
}>();

const host = ref<HTMLDivElement | null>(null);
const err = ref("");
const status = ref("加载 URDF…");
const mode = ref<"visual" | "collision">("visual");
const envId = ref("");
const missing = ref<string[]>([]);
const meshCount = ref(0);
const jointValues = ref<Record<string, number>>({});

const movable = () =>
  (props.preview?.tree || []).filter((j: any) => {
    const t = String(j.type || "").toLowerCase();
    return t && t !== "fixed";
  });

function lim(j: any) {
  const lo = Number(j.limit?.lower ?? -3.14);
  const hi = Number(j.limit?.upper ?? 3.14);
  return { lo: Number.isFinite(lo) ? lo : -3.14, hi: Number.isFinite(hi) ? hi : 3.14 };
}

let renderer: any = null;
let scene: any = null;
let camera: any = null;
let controls: any = null;
let robot: any = null;
let raf = 0;
let ro: ResizeObserver | null = null;
let THREE: any = null;

const published = () => (props.envs || []).filter((e) => e.status === "active");

function assetFetch(url: string) {
  return fetch(url, { credentials: "include", signal: AbortSignal.timeout(60000) });
}

function rewriteMeshes(xml: string) {
  const byFile = new Map<string, string>();
  for (const m of props.preview?.meshes || []) {
    if (m.filename) byFile.set(m.filename, m.url);
    if (m.path) byFile.set(m.path, m.url);
    const base = String(m.filename || m.path || "").split("/").pop();
    if (base && m.url) byFile.set(base, m.url);
  }
  return xml.replace(/filename=(['"])([^'"]+)\1/gi, (_all, q, name) => {
    const base = String(name).split("/").pop() || "";
    const url = byFile.get(name) || byFile.get(base);
    if (!url) return `filename=${q}${name}${q}`;
    return `filename=${q}${url}${q}`;
  });
}

function wireBox(THREELib: any) {
  const g = new THREELib.BoxGeometry(0.08, 0.08, 0.08);
  const e = new THREELib.EdgesGeometry(g);
  const line = new THREELib.LineSegments(
    e,
    new THREELib.LineBasicMaterial({ color: 0x53d4ff }),
  );
  g.dispose();
  return line;
}

function cloneMat(obj: any) {
  if (!obj.material || obj.userData._ebMatCloned) return;
  obj.material = Array.isArray(obj.material)
    ? obj.material.map((m: any) => m.clone())
    : obj.material.clone();
  obj.userData._ebMatCloned = true;
}

function wireCollision() {
  if (!robot || !THREE) return;
  robot.traverse((obj: any) => {
    if (!obj.isURDFCollider) return;
    obj.traverse((child: any) => {
      if (!child.isMesh || !child.material) return;
      cloneMat(child);
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const mat of mats) {
        mat.wireframe = true;
        mat.color?.set?.(0x53d4ff);
        mat.emissive?.set?.(0x102028);
        mat.transparent = true;
        mat.opacity = 0.95;
        mat.needsUpdate = true;
      }
    });
  });
}

function applyMode() {
  if (!robot) return;
  robot.traverse((obj: any) => {
    const isCol = obj.isURDFCollider || obj.type === "URDFCollider";
    const isVis = obj.isURDFVisual || obj.type === "URDFVisual";
    if (isCol) obj.visible = mode.value === "collision";
    if (isVis) obj.visible = mode.value === "visual";
  });
}

function applyJoints() {
  if (!robot) return;
  for (const [name, val] of Object.entries(jointValues.value)) {
    try {
      if (typeof robot.setJointValue === "function") robot.setJointValue(name, val);
      else robot.joints?.[name]?.setJointValue?.(val);
    } catch {
      /* ignore unknown joint */
    }
  }
}

function applyHighlight() {
  if (!robot) return;
  const env = (props.envs || []).find((e) => e.id === envId.value);
  const order: string[] | undefined = env?.spec?.joint_order;
  const active = order && order.length ? new Set(order) : null;
  const fadedLinks = new Set<string>();
  if (active && robot.joints) {
    for (const joint of Object.values<any>(robot.joints)) {
      if (active.has(joint.name) || active.has(joint.urdfName)) continue;
      if (joint.jointType === "fixed") continue;
      joint.traverse((obj: any) => {
        if (obj.isURDFLink) fadedLinks.add(obj.name);
      });
    }
  }
  robot.traverse((obj: any) => {
    if (!obj.isMesh || !obj.material) return;
    cloneMat(obj);
    let link: any = obj;
    while (link && !link.isURDFLink && link.parent) link = link.parent;
    const dim = fadedLinks.has(link?.name);
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of mats) {
      if (mat.wireframe) {
        mat.opacity = dim ? 0.2 : 0.95;
      } else {
        mat.transparent = true;
        mat.opacity = dim ? 0.18 : mode.value === "visual" ? 0.98 : 1;
      }
      mat.needsUpdate = true;
    }
  });
}

function fit(THREELib: any) {
  if (!robot || !camera || !controls) return;
  const box = new THREELib.Box3().setFromObject(robot);
  const size = box.getSize(new THREELib.Vector3());
  const center = box.getCenter(new THREELib.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 0.4);
  camera.near = Math.max(radius / 200, 0.01);
  camera.far = radius * 40;
  camera.updateProjectionMatrix();
  camera.position.set(center.x + radius * 1.7, center.y + radius * 1.05, center.z + radius * 1.7);
  controls.target.copy(center);
  controls.update();
}

function resize() {
  if (!host.value || !renderer || !camera) return;
  const w = host.value.clientWidth || 640;
  const h = host.value.clientHeight || 420;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function loop() {
  raf = requestAnimationFrame(loop);
  controls?.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

async function boot() {
  err.value = "";
  missing.value = [...(props.preview?.missing_meshes || [])];
  meshCount.value = 0;
  const url = props.preview?.urdf_url;
  await nextTick();
  if (!url || !host.value) {
    status.value = "没有可加载的 URDF";
    return;
  }
  status.value = "加载 three.js…";
  THREE = await import("three");
  const [{ OrbitControls }, { STLLoader }, URDFLoaderMod] = await Promise.all([
    import("three/examples/jsm/controls/OrbitControls.js"),
    import("three/examples/jsm/loaders/STLLoader.js"),
    import("urdf-loader"),
  ]);
  const URDFLoader = (URDFLoaderMod as any).default || URDFLoaderMod;
  const w = host.value.clientWidth || 640;
  const h = host.value.clientHeight || 420;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x071018);
  camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 50);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setSize(w, h);
  host.value.innerHTML = "";
  host.value.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x1a2a33, 1.15));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(2.2, 4.2, 3.1);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x53d4ff, 0.25);
  fill.position.set(-2, 1, -1);
  scene.add(fill);
  const grid = new THREE.GridHelper(2, 16, 0x284052, 0x1d3243);
  scene.add(grid);

  status.value = "拉取 URDF + STL…";
  const xml = await assetFetch(url).then(async (r) => {
    if (!r.ok) throw new Error(`URDF ${r.status}`);
    return r.text();
  });
  const rewritten = rewriteMeshes(xml);
  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  loader.parseCollision = true;
  loader.parseVisual = true;
  loader.workingPath = "";
  let pending = 0;
  let loaded = 0;
  const waitMeshes = new Promise<void>((resolve) => {
    const maybe = () => {
      if (pending === 0) resolve();
    };
    loader.loadMeshCb = (path: string, _mgr: any, material: any, onComplete: (obj: any, e?: Error) => void) => {
      pending += 1;
      const stl = new STLLoader();
      assetFetch(path)
        .then(async (r) => {
          if (!r.ok) throw new Error(`mesh ${r.status}`);
          return r.arrayBuffer();
        })
        .then((buf) => {
          const geom = stl.parse(buf);
          const fallback = new THREE.MeshStandardMaterial({
            color: 0x9ec9d8,
            metalness: 0.18,
            roughness: 0.52,
          });
          const mesh = new THREE.Mesh(geom, material || fallback);
          loaded += 1;
          meshCount.value = loaded;
          onComplete(mesh);
        })
        .catch(() => {
          missing.value = [...new Set([...missing.value, path])];
          onComplete(wireBox(THREE));
        })
        .finally(() => {
          pending -= 1;
          maybe();
        });
    };
    robot = loader.parse(rewritten, "");
    queueMicrotask(maybe);
  });
  await waitMeshes;
  if (!robot) throw new Error("URDF 解析失败");
  scene.add(robot);
  robot.rotation.x = -Math.PI / 2;
  const init: Record<string, number> = {};
  for (const j of movable()) init[j.name] = 0;
  jointValues.value = init;
  wireCollision();
  applyMode();
  applyHighlight();
  applyJoints();
  fit(THREE);
  status.value = "";
  loop();
  ro = new ResizeObserver(resize);
  ro.observe(host.value);
}

function teardown() {
  cancelAnimationFrame(raf);
  raf = 0;
  ro?.disconnect();
  ro = null;
  controls?.dispose();
  renderer?.dispose();
  if (renderer?.domElement?.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  renderer = scene = camera = controls = robot = null;
}

watch(mode, () => {
  applyMode();
  applyHighlight();
});
watch(envId, applyHighlight);
watch(jointValues, applyJoints, { deep: true });

onMounted(async () => {
  try {
    await boot();
  } catch (e: any) {
    err.value = e.message || String(e);
    status.value = "";
  }
});
onUnmounted(teardown);
</script>
<template>
  <div>
    <div class="row" style="margin-bottom:.55rem">
      <button type="button" :class="{ primary: mode === 'visual' }" @click="mode = 'visual'">visual</button>
      <button type="button" :class="{ primary: mode === 'collision' }" @click="mode = 'collision'">collision</button>
      <select v-if="published().length" v-model="envId" style="margin-left:auto;max-width:280px">
        <option value="">全部关节</option>
        <option v-for="e in published()" :key="e.id" :value="e.id">{{ e.id }}</option>
      </select>
    </div>
    <div ref="host" class="urdf-stage" />
    <p class="muted xs" v-if="status">{{ status }}</p>
    <p class="muted xs" v-else-if="meshCount">STL {{ meshCount }} 件 · 拖拽旋转 · 滚轮缩放</p>
    <p class="error xs" v-if="err">{{ err }}</p>
    <p class="warn xs" v-if="missing.length">缺 mesh {{ missing.length }}：用 collision 盒线框代替，不用纯色块冒充。</p>
    <div class="joint-sliders" v-if="movable().length">
      <label v-for="j in movable()" :key="j.name">
        <span class="mono">{{ j.name }}</span>
        <input
          type="range"
          :min="lim(j).lo"
          :max="lim(j).hi"
          step="0.01"
          v-model.number="jointValues[j.name]"
        />
        <span>{{ Number(jointValues[j.name] || 0).toFixed(2) }}</span>
      </label>
    </div>
  </div>
</template>
