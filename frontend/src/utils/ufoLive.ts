export type LiveIdentity = { run_id: string; model_id: string; payload_sha256: string; plan_hash: string };
export type LiveState = LiveIdentity & {
  schema_version: string; seq: number; status: string; task: string; episode: number; step: number;
  sim_seconds: number; wall_seconds: number; updated_at: number; realtime_factor: number;
  body_poses: number[][]; root_position: number[]; velocity: number[]; height: number; tilt: number;
  limits: { joint: number; velocity: number; torque: number }; inference_ms: number;
  ack: { id: string; action: string; applied_step: number; task: string; episode: number } | null; error?: string;
};
export type LiveCamera = { azimuth: number; elevation: number; distance: number };
export const defaultLiveCamera = (): LiveCamera => ({ azimuth: 135, elevation: -18, distance: 3 });
export type LiveFrameMetadata = {
  seq: number; step: number; episode: number; updated_at: number; rendered_at: number; camera_seq: number;
  width: number; height: number; render_ms: number; encode_ms: number;
};
export type LiveFrame = { metadata: LiveFrameMetadata; jpeg: Uint8Array };
const vector = (value: unknown, length: number): value is number[] => Array.isArray(value) && value.length === length && value.every(Number.isFinite);

export function liveId(): string {
  const crypto = globalThis.crypto;
  if (typeof crypto?.randomUUID === "function") return crypto.randomUUID();
  // getRandomValues remains available on HTTP origins where randomUUID is unavailable.
  if (typeof crypto?.getRandomValues !== "function") throw Error("当前浏览器不支持安全随机数，请更新浏览器。");
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

export function liveStreamUrl(runId: string, location: Pick<Location, "protocol" | "host"> = window.location): string {
  return `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/v1/ufo-native/live/runs/${encodeURIComponent(runId)}/stream`;
}

export function validateLiveState(value: LiveState, expected: LiveIdentity, previous: LiveState | null): LiveState | null {
  if (value?.schema_version !== "eb/ufo-live-state/v1" || Object.keys(expected).some(key => value[key as keyof LiveIdentity] !== expected[key as keyof LiveIdentity])) throw Error("实时状态身份与冻结模型/运行不一致");
  if (!Number.isSafeInteger(value.seq) || value.seq < 0 || !Number.isFinite(value.updated_at)) throw Error("实时状态序号或时间无效");
  if (previous && value.seq <= previous.seq) return null;
  const emptyStartup = ["initializing", "error"].includes(value.status) && Array.isArray(value.body_poses) && value.body_poses.length === 0;
  if ((!emptyStartup && (!Array.isArray(value.body_poses) || value.body_poses.length !== 24 || !value.body_poses.every(pose => vector(pose, 7) && Math.hypot(...pose.slice(3)) > .99 && Math.hypot(...pose.slice(3)) < 1.01)))
    || !vector(value.root_position, 3) || !vector(value.velocity, 2)
    || ![value.height, value.tilt, value.inference_ms, value.realtime_factor, value.step, value.episode, value.sim_seconds, value.wall_seconds, value.limits?.joint, value.limits?.velocity, value.limits?.torque].every(Number.isFinite)
    || !["initializing", "running", "paused", "fallen", "finalizing", "finished", "error"].includes(value.status) || typeof value.task !== "string") throw Error("实时状态包含无效姿态或指标");
  return value;
}

export function parseLiveFrame(data: ArrayBuffer): LiveFrame {
  if (data.byteLength < 8 || data.byteLength > 4 + 4096 + 2 * 1024 * 1024) throw Error("实时图像帧长度无效");
  const length = new DataView(data).getUint32(0, false);
  if (!length || length > 4096 || length > data.byteLength - 8) throw Error("实时图像元数据长度无效");
  const jpeg = new Uint8Array(data, 4 + length);
  if (jpeg.byteLength > 2 * 1024 * 1024 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8 || jpeg[jpeg.length - 2] !== 0xff || jpeg[jpeg.length - 1] !== 0xd9) throw Error("实时图像不是有效的有界 JPEG");
  const metadata = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(data, 4, length))) as LiveFrameMetadata;
  if (!metadata || ![metadata.seq, metadata.step, metadata.episode, metadata.camera_seq].every(value => Number.isSafeInteger(value) && value >= 0)
    || metadata.width !== 640 || metadata.height !== 480
    || ![metadata.updated_at, metadata.rendered_at, metadata.render_ms, metadata.encode_ms].every(value => Number.isFinite(value) && value >= 0)) throw Error("实时图像序号、尺寸或时间无效");
  return { metadata, jpeg };
}

type DecodedFrame = { width: number; height: number; close(): void };
export function createLiveFrameSink<T extends DecodedFrame>(options: {
  decode(frame: LiveFrame): Promise<T>;
  draw(bitmap: T, metadata: LiveFrameMetadata, receivedAt: number): void;
  error(error: Error): void;
  now?: () => number;
}) {
  let busy = false, disposed = false, generation = 0, highestSeq = -1;
  let latest: { frame: LiveFrame; acknowledge: (seq: number) => void; receivedAt: number; generation: number } | null = null;
  async function drain() {
    if (busy || !latest || disposed) return;
    const item = latest; latest = null; busy = true;
    let bitmap: T | undefined;
    try {
      bitmap = await options.decode(item.frame);
      if (disposed || item.generation !== generation) return;
      if (bitmap.width !== item.frame.metadata.width || bitmap.height !== item.frame.metadata.height) throw Error("JPEG 解码尺寸与元数据不一致");
      options.draw(bitmap, item.frame.metadata, item.receivedAt);
      item.acknowledge(item.frame.metadata.seq);
    } catch (error) {
      if (!disposed && item.generation === generation) { latest = null; generation++; options.error(error instanceof Error ? error : Error(String(error))); }
    } finally {
      bitmap?.close(); busy = false;
      if (latest && !disposed) void drain();
    }
  }
  return {
    push(data: ArrayBuffer, acknowledge: (seq: number) => void) {
      if (disposed) return;
      try {
        const frame = parseLiveFrame(data);
        if (frame.metadata.seq <= highestSeq) throw Error("实时图像序号没有递增");
        highestSeq = frame.metadata.seq;
        latest = { frame, acknowledge, receivedAt: (options.now || (() => performance.now()))(), generation };
        void drain();
      } catch (error) { latest = null; generation++; options.error(error instanceof Error ? error : Error(String(error))); }
    },
    reset() { generation++; latest = null; highestSeq = -1; },
    dispose() { disposed = true; generation++; latest = null; },
  };
}
