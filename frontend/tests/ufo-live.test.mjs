import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

function helpers(crypto = { randomUUID: () => 'random-id' }) {
  const context = { exports: {}, crypto, TextDecoder, Uint8Array, DataView, performance };
  vm.runInNewContext(ts.transpile(readFileSync(new URL('../src/utils/ufoLive.ts', import.meta.url), 'utf8'), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS,
  }), context);
  return context.exports;
}
const live = helpers();
const identity = { run_id: 'run_live', model_id: 'mv_live', payload_sha256: 'payload', plan_hash: 'plan' };
const snapshot = (seq = 1, changes = {}) => ({ ...identity, schema_version: 'eb/ufo-live-state/v1', seq,
  status: 'running', task: 'move-ego-0-0', episode: 0, step: seq, sim_seconds: .02 * seq,
  wall_seconds: .02 * seq, updated_at: 100, realtime_factor: 1, root_position: [0, 0, .8],
  body_poses: Array.from({ length: 24 }, () => [0, 0, .8, 1, 0, 0, 0]),
  velocity: [0, 0], height: .8, tilt: 0, limits: { joint: 0, velocity: 0, torque: 0 }, inference_ms: 2, ack: null, ...changes });
const caps = { image_ready: true, models: [{ id: 'mv_live', name: 'Frozen', embodiment_env_id: 'ee' }],
  environments: [{ id: 'env_live', name: 'GPU', backend: 'mjlab', embodiment_env_id: 'ee' }],
  tasks: [{ id: 'move-ego-0-0', label: '站立', speed: 0 }, { id: 'move-ego-0-0.3', label: '0.3', speed: .3 }] };
function setup(respond = async () => ({}), query = {}, cryptoApi, fetchRespond) {
  const calls = [], fallbacks = [], timers = new Map(), watchers = [], sockets = [], receivedFrames = [], listeners = new Map();
  let cleanup, mounted, leaveGuard, updateGuard, timerId = 0, uuid = 0, now = 100_000, resets = 0;
  const browserCrypto = cryptoApi || { randomUUID: () => `command-${++uuid}` };
  class Socket {
    static OPEN = 1;
    constructor(url) { this.url = url; this.readyState = 0; this.sent = []; sockets.push(this); }
    open() { this.readyState = 1; this.onopen?.(); }
    send(text) { if (this.fail) throw Error('connection lost'); this.sent.push(JSON.parse(text)); }
    message(data) { this.onmessage?.({ data: typeof data === 'string' || data instanceof ArrayBuffer ? data : JSON.stringify(data) }); }
    close() { if (this.failClose) throw Error('socket close failed'); this.readyState = 3; this.onclose?.(); }
  }
  const source = readFileSync(new URL('../src/views/SimulationValidation.vue', import.meta.url), 'utf8').match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ...helpers(browserCrypto), ref, computed, AbortController, ArrayBuffer,
    performance: { now: () => now }, WebSocket: Socket, window: { location: { protocol: 'http:', host: '192.168.1.2:8000' },
      addEventListener(name, handler) { listeners.set(name, handler); }, removeEventListener(name) { listeners.delete(name); } },
    // The imported URL helper receives its own VM global, so bind the browser location explicitly.
    liveStreamUrl: run => live.liveStreamUrl(run, { protocol: 'http:', host: '192.168.1.2:8000' }),
    useRoute: () => ({ query }), onMounted(fn) { mounted = fn; }, onUnmounted(fn) { cleanup = fn; }, watch(...args) { watchers.push(args); },
    onBeforeRouteLeave(fn) { leaveGuard = fn; }, onBeforeRouteUpdate(fn) { updateGuard = fn; },
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; }, clearTimeout(id) { timers.delete(id); },
    api: async (path, init) => { calls.push([path, init]); return respond(path, init); },
    fetch: async (path, init) => {
      fallbacks.push([path, init]);
      const result = fetchRespond ? await fetchRespond(path, init) : path.endsWith('/close-pending') ? { closed: true, job: null } : { id: path.split('/').at(-2), status: 'cancelled' };
      return { ok: !result.httpError, status: result.httpError || 200, json: async () => result, text: async () => result.message || 'close failed' };
    },
  });
  const code = ts.transpile(source.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.subject = { load, check, start, pollJob, heartbeat, command, retryCommand, reconnect, openStream, cameraChanged, modelId, environmentId, preview, job, state, stage, pending, activeTask, stale, canControl, error, transportError, connected, commandUncertain, clock: now,
    closeEnvironment: typeof closeEnvironment === 'function' ? closeEnvironment : null,
    closeOtherSession: typeof closeOtherSession === 'function' ? closeOtherSession : null,
    activeSessions: typeof activeSessions === 'object' ? activeSessions : null,
    closing: typeof closing === 'object' ? closing : null, busy,
    closeError: typeof closeError === 'object' ? closeError : null,
    closeNotice: typeof closeNotice === 'object' ? closeNotice : null,
    otherSessions: typeof otherSessions === 'object' ? otherSessions : null };`, context);
  const s = context.subject;
  s.stage.value = { reset() { resets++; }, receiveFrame(data, acknowledge) { receivedFrames.push(data); acknowledge(9); } };
  return { ...s, calls, fallbacks, sockets, timers, watchers, receivedFrames, listeners, get resets() { return resets; },
    now(value) { now = value; s.clock.value = value; }, mount: () => mounted(), cleanup: () => cleanup(),
    leave: (...args) => leaveGuard(...args), update: (...args) => updateGuard(...args), event: (name, value = {}) => listeners.get(name)?.(value),
    connect() { s.job.value = { id: 'job_live', status: 'running', ...identity }; s.openStream(); sockets.at(-1).open(); sockets.at(-1).message({ type: 'state', state: snapshot() }); return sockets.at(-1); },
    async tick(ms) { const next = [...timers].find(([, value]) => value.ms === ms); assert.ok(next, `expected ${ms}ms timer`); timers.delete(next[0]); await next[1].fn(); },
  };
}
const flush = async () => { await new Promise(resolve => setImmediate(resolve)); };
function frame(changes = {}, jpeg = Uint8Array.from([255, 216, 255, 217])) {
  const metadata = { seq: 1, step: 10, episode: 0, camera_seq: 1, updated_at: 100, rendered_at: 100.01, width: 640, height: 480, render_ms: 2, encode_ms: 1, ...changes };
  const json = new TextEncoder().encode(JSON.stringify(metadata));
  const result = new Uint8Array(4 + json.length + jpeg.length);
  new DataView(result.buffer).setUint32(0, json.length, false); result.set(json, 4); result.set(jpeg, 4 + json.length); return result.buffer;
}

test('same-origin WS URLs follow HTTP/HTTPS without credentials, and HTTP crypto fallback remains usable', async () => {
  assert.equal(live.liveStreamUrl('run 1', { protocol: 'http:', host: '10.0.0.1:8000' }), 'ws://10.0.0.1:8000/api/v1/ufo-native/live/runs/run%201/stream');
  assert.equal(live.liveStreamUrl('run', { protocol: 'https:', host: 'example.test' }), 'wss://example.test/api/v1/ufo-native/live/runs/run/stream');
  let nonce = 0;
  const s = setup(async path => path.endsWith('/capabilities') ? caps : { plan: { hash: 'plan' } }, {}, { getRandomValues(bytes) { bytes.fill(0); bytes[15] = ++nonce; return bytes; } });
  await s.load(); await s.check(); const key = s.preview.value.key; assert.match(key, /^[a-f0-9]{32}$/);
  const ws = s.connect(); s.command('task', 'move-ego-0-0.3'); const command = ws.sent.find(message => message.type === 'control').command;
  assert.match(command.id, /^[a-f0-9]{32}$/); s.cleanup();
  assert.equal(ws.sent.filter(message => message.type === 'control').length, 1, 'resource cleanup does not send WS stop');
  assert.equal(new Set([key, command.id]).size, 2);
  assert.equal(s.fallbacks[0][0], '/api/v1/ufo-native/live/sessions/job_live/close');
});

test('closing a queued environment uses the resource-close API without requiring a Run or socket', async () => {
  const s = setup(async path => ({ id: 'job_queued', status: path.endsWith('/close') ? 'cancelled' : 'queued' }));
  s.job.value = { id: 'job_queued', status: 'queued' };
  assert.equal(typeof s.closeEnvironment, 'function');
  assert.equal(await s.closeEnvironment(), true);
  assert.equal(s.job.value.status, 'cancelled');
  assert.equal(s.calls.filter(([path]) => path.endsWith('/sessions/job_queued/close')).length, 1);
  assert.equal(s.sockets.length, 0); s.cleanup();
});

test('start freezes standing default once; task changes remain in the same Run and await applied state ack', async () => {
  const s = setup(async path => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions')) return { id: 'job_live', status: 'queued' };
    return { id: 'job_live', status: 'running', ...identity };
  });
  await s.load(); await s.start(); await flush(); assert.equal(s.sockets.length, 1);
  const ws = s.sockets[0]; ws.open(); ws.message({ type: 'state', state: snapshot() });
  assert.equal(s.calls.filter(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST').length, 1);
  assert.equal(JSON.parse(s.calls.find(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST')[1].body).initial_task, 'move-ego-0-0');
  s.command('task', 'move-ego-0-0.3'); const id = s.pending.value.id;
  assert.equal(s.activeTask.value, 'move-ego-0-0'); assert.equal(s.canControl.value, false);
  ws.message({ type: 'control_result', id, ok: true, status: 200 });
  assert.equal(s.pending.value.id, id, 'accepted is not yet applied');
  ws.message({ type: 'state', state: snapshot(2, { ack: { id: 'other', action: 'task' } }) }); assert.equal(s.pending.value.id, id);
  ws.message({ type: 'state', state: snapshot(3, { task: 'move-ego-0-0.3', ack: { id, action: 'task', applied_step: 3, episode: 0, task: 'move-ego-0-0.3' } }) });
  assert.equal(s.pending.value, null); assert.equal(s.activeTask.value, 'move-ego-0-0.3');
  assert.equal(s.job.value.run_id, identity.run_id); assert.equal(s.state.value.episode, 0);
  assert.equal(ws.sent.filter(message => message.type === 'control').length, 1); s.cleanup();
});

test('preview retries reuse idempotency and a queued Job opens no socket until run_id exists', async () => {
  let fail = true;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') { if (fail) { fail = false; throw Error('temporary'); } return { id: 'job_live', status: 'queued' }; }
    return { id: 'job_live', status: 'queued' };
  });
  await s.load(); await s.check(); await s.start(); await s.start(); await flush();
  const writes = s.calls.filter(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST');
  assert.equal(writes.length, 2); assert.equal(writes[0][1].headers['Idempotency-Key'], writes[1][1].headers['Idempotency-Key']);
  assert.equal(s.sockets.length, 0); assert.ok([...s.timers.values()].some(timer => timer.ms === 2000)); s.cleanup(); assert.equal(s.timers.size, 0);
});

test('state freshness uses monotonic receive time; stale and invalid states disable control', () => {
  const s = setup(); const ws = s.connect(); assert.equal(s.canControl.value, true);
  // Server clock can differ; it must not invalidate a newly received frame.
  ws.message({ type: 'state', state: snapshot(2, { updated_at: 1 }) }); assert.equal(s.canControl.value, true);
  s.now(102000); assert.equal(s.stale.value, true); assert.equal(s.canControl.value, false);
  ws.message({ type: 'state', state: snapshot(3) }); assert.equal(s.canControl.value, true);
  ws.message({ type: 'state', state: snapshot(2) }); assert.equal(s.state.value.seq, 3);
  ws.message({ type: 'state', state: snapshot(4, { model_id: 'other' }) });
  assert.equal(s.state.value.seq, 3); assert.equal(s.connected.value, false); assert.match(s.transportError.value, /断开/); s.cleanup();
  assert.throws(() => live.validateLiveState(snapshot(1, { body_poses: [[Infinity, 0, 0, 1, 0, 0, 0]] }), identity, null));
});

test('heartbeat, frame acknowledgement and controls share one socket; unmount uses independent HTTP resource close', async () => {
  const s = setup(); const ws = s.connect(); s.command('pause'); const id = s.pending.value.id;
  await s.tick(3000); assert.equal(s.pending.value.id, id); assert.equal(ws.sent.filter(m => m.type === 'heartbeat').length, 2);
  ws.message(frame()); assert.equal(s.receivedFrames.length, 1); assert.equal(ws.sent.at(-1).type, 'frame_ack');
  assert.equal(s.calls.length, 0, 'no state/scene/control/heartbeat HTTP polling');
  s.cleanup(); assert.equal(s.timers.size, 0); assert.equal(ws.readyState, 3);
  assert.equal(ws.sent.filter(message => message.type === 'control').length, 1);
  assert.equal(s.fallbacks[0][1].keepalive, true);
});

test('uncertain sends retry the same ID and socket reconnection never creates a new session', async () => {
  const s = setup(); let ws = s.connect(); ws.fail = true; s.command('pause'); const id = s.pending.value.id;
  assert.equal(s.commandUncertain.value, true); ws.fail = false; s.retryCommand(); assert.equal(ws.sent.at(-1).command.id, id);
  ws.close(); assert.equal(s.canControl.value, false); await s.tick(500); ws = s.sockets.at(-1); ws.open();
  ws.message({ type: 'state', state: snapshot(2) }); s.retryCommand(); assert.equal(ws.sent.at(-1).command.id, id);
  ws.message({ type: 'control_result', id, ok: true, status: 200 }); assert.equal(s.pending.value.id, id);
  ws.message({ type: 'state', state: snapshot(3, { status: 'paused', ack: { id, action: 'pause', applied_step: 3 } }) });
  assert.equal(s.pending.value, null); assert.equal(s.job.value.run_id, identity.run_id); assert.equal(s.calls.length, 0); s.cleanup();
});

test('definite control rejection clears pending; missing applied acknowledgement offers same-ID retry', async () => {
  const s = setup(); const ws = s.connect(); s.command('pause');
  ws.message({ type: 'control_result', id: s.pending.value.id, ok: false, status: 409, detail: 'command slot busy' });
  assert.equal(s.pending.value, null); assert.equal(s.canControl.value, true); assert.match(s.error.value, /command slot busy/);
  s.command('pause'); const id = s.pending.value.id; await s.tick(5000); assert.equal(s.commandUncertain.value, true);
  s.retryCommand(); assert.equal(ws.sent.at(-1).command.id, id); s.cleanup();
});

test('reconnect attempts have bounded exponential backoff and do not revive after cleanup', async () => {
  const s = setup(); s.connect().close();
  for (const ms of [500, 1000, 2000, 4000, 5000, 5000, 5000, 5000]) { await s.tick(ms); s.sockets.at(-1).close(); }
  assert.equal(s.sockets.length, 9); assert.equal(s.timers.size, 0); assert.match(s.transportError.value, /重连未成功/);
  s.reconnect(); assert.equal(s.sockets.length, 10); s.cleanup(); assert.equal(s.timers.size, 0);
});

test('lease renewal failure is visible and disables controls before reconnection', () => {
  const s = setup(); const ws = s.connect();
  ws.message({ type: 'heartbeat_result', ok: false, status: 409, detail: 'worker ended' });
  assert.equal(s.connected.value, false); assert.equal(s.canControl.value, false);
  assert.match(s.transportError.value, /续租失败.*worker ended/); s.cleanup();
});

test('camera changes coalesce at 30Hz and remain available while physics is paused', async () => {
  const s = setup(); const ws = s.connect();
  assert.deepEqual(ws.sent.find(m => m.type === 'camera'), { type: 'camera', seq: 1, azimuth: 135, elevation: -18, distance: 3 });
  ws.message({ type: 'state', state: snapshot(2, { status: 'paused' }) });
  s.cameraChanged({ azimuth: 140, elevation: -20, distance: 3 }); s.cameraChanged({ azimuth: 145, elevation: -21, distance: 2 });
  assert.equal(ws.sent.filter(m => m.type === 'camera').length, 1); await s.tick(34);
  assert.deepEqual(ws.sent.at(-1), { type: 'camera', seq: 2, azimuth: 145, elevation: -21, distance: 2 }); s.cleanup();
});

test('finalizing stops heartbeat but metadata polling continues until artifact collection', async () => {
  const s = setup(async () => ({ id: 'job_live', status: 'succeeded', files: [{ path: 'live.mp4', url: '/video' }], ...identity }));
  const ws = s.connect(); ws.message({ type: 'state', state: snapshot(2, { status: 'finalizing' }) });
  assert.equal([...s.timers.values()].some(t => t.ms === 3000), false); assert.equal(s.canControl.value, false);
  await s.pollJob(); assert.equal(s.job.value.files[0].path, 'live.mp4'); assert.equal(s.timers.size, 0); s.cleanup();
});

test('initializing and error snapshots may omit poses; active snapshots require physical bodies', () => {
  assert.equal(live.validateLiveState(snapshot(1, { status: 'initializing', body_poses: [] }), identity, null).status, 'initializing');
  assert.equal(live.validateLiveState(snapshot(2, { status: 'error', body_poses: [] }), identity, null).status, 'error');
  assert.throws(() => live.validateLiveState(snapshot(3, { body_poses: [] }), identity, null));
});

test('binary frame parser enforces bounded metadata/JPEG, finite times, dimensions and integer sequence', () => {
  const parsed = live.parseLiveFrame(frame()); assert.equal(parsed.metadata.width, 640); assert.equal(parsed.jpeg.length, 4);
  for (const change of [{ seq: -1 }, { seq: 1.1 }, { seq: Number.MAX_SAFE_INTEGER + 1 }, { camera_seq: -1 }, { step: null }, { updated_at: null }, { rendered_at: null }, { render_ms: -1 }, { encode_ms: null }, { width: 0 }, { width: 6400 }, { height: 481 }]) assert.throws(() => live.parseLiveFrame(frame(change)), JSON.stringify(change));
  for (const length of [0, 4097, 0xffffffff]) { const bad = frame(); new DataView(bad).setUint32(0, length, false); assert.throws(() => live.parseLiveFrame(bad)); }
  assert.throws(() => live.parseLiveFrame(new ArrayBuffer(4)));
  assert.throws(() => live.parseLiveFrame(frame({}, new Uint8Array(2 * 1024 * 1024 + 1))));
  assert.throws(() => live.parseLiveFrame(frame({}, Uint8Array.from([1, 2, 3, 4]))));
  const badUtf8 = frame(); new Uint8Array(badUtf8)[4] = 255; assert.throws(() => live.parseLiveFrame(badUtf8));
});

test('JPEG sink has only one decoder and one latest pending frame; ACK follows draw and bitmaps close', async () => {
  const decodes = [], drawn = [], acks = [], closed = [], errors = [];
  const sink = live.createLiveFrameSink({ decode(frame) { return new Promise(resolve => decodes.push({ seq: frame.metadata.seq, resolve })); },
    draw(_bitmap, metadata) { drawn.push(metadata.seq); }, error(error) { errors.push(error); }, now: () => 10 });
  const complete = item => item.resolve({ width: 640, height: 480, close() { closed.push(item.seq); } });
  sink.push(frame({ seq: 1 }), seq => acks.push(seq)); sink.push(frame({ seq: 2 }), seq => acks.push(seq)); sink.push(frame({ seq: 3 }), seq => acks.push(seq));
  assert.equal(decodes.length, 1); assert.deepEqual(acks, []); complete(decodes[0]); await flush();
  assert.deepEqual(drawn, [1]); assert.deepEqual(acks, [1]); assert.equal(decodes.length, 2); assert.equal(decodes[1].seq, 3);
  complete(decodes[1]); await flush(); assert.deepEqual(acks, [1, 3]); assert.deepEqual(closed, [1, 3]);
  sink.push(frame({ seq: 3 }), () => {}); assert.equal(errors.length, 1); sink.dispose();
});

test('reset/unmount closes outstanding decoded images without ACK; invalid sizes and decode failure are visible', async () => {
  let resolve; const acks = [], drawn = [], errors = [], closed = [];
  const sink = live.createLiveFrameSink({ decode() { return new Promise(done => { resolve = done; }); }, draw() { drawn.push(1); }, error(error) { errors.push(error.message); } });
  sink.push(frame(), seq => acks.push(seq)); sink.reset();
  resolve({ width: 640, height: 480, close() { closed.push(1); } }); await flush(); assert.deepEqual(acks, []); assert.deepEqual(drawn, []);
  sink.push(frame(), seq => acks.push(seq)); sink.dispose(); resolve({ width: 640, height: 480, close() { closed.push(2); } }); await flush(); assert.deepEqual(acks, []); assert.deepEqual(closed, [1, 2]);
  const mismatch = live.createLiveFrameSink({ decode: async () => ({ width: 1, height: 1, close() { closed.push(3); } }), draw() { drawn.push(1); }, error(error) { errors.push(error.message); } });
  mismatch.push(frame(), seq => acks.push(seq)); await flush(); assert.match(errors.at(-1), /尺寸/); assert.deepEqual(acks, []); assert.equal(closed.at(-1), 3);
  const failed = live.createLiveFrameSink({ decode: async () => { throw Error('bad JPEG'); }, draw() {}, error(error) { errors.push(error.message); } });
  failed.push(frame(), () => acks.push(1)); await flush(); assert.match(errors.at(-1), /bad JPEG/); assert.deepEqual(acks, []);
});

function stageSetup() {
  const source = readFileSync(new URL('../src/components/NativeSimulationStage.vue', import.meta.url), 'utf8').match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  let mounted, cleanup; const timers = new Map(), events = [], drawn = [], closed = [];
  const context = vm.createContext({ ...live, ref, Blob, performance: { now: () => 100 },
    onMounted(fn) { mounted = fn; }, onUnmounted(fn) { cleanup = fn; }, defineEmits: () => (...args) => events.push(args), defineExpose() {},
    createImageBitmap: async blob => { assert.equal(blob.type, 'image/jpeg'); return { width: 640, height: 480, close() { closed.push(1); } }; },
    setTimeout(fn, ms) { const id = timers.size + 1; timers.set(id, { fn, ms }); return id; }, clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(`${ts.transpile(source.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None })}\nglobalThis.subject = { canvas, receiveFrame, reset, pointerDown, pointerMove, pointerUp, wheel, keydown, updateCamera, camera, metadata };`, context);
  context.subject.canvas.value = { getContext: () => ({ drawImage(...args) { drawn.push(args); }, clearRect() {} }) };
  return { ...context.subject, events, drawn, closed, timers, mount: () => mounted(), cleanup: () => cleanup() };
}

test('native stage decodes JPEG into canvas, exposes frame metadata and releases bitmap/timer', async () => {
  const stage = stageSetup(), acks = []; stage.mount(); stage.receiveFrame(frame(), seq => acks.push(seq)); await flush();
  assert.equal(stage.drawn.length, 1); assert.deepEqual(acks, [1]); assert.equal(stage.metadata.value.camera_seq, 1); assert.equal(stage.closed.length, 1);
  stage.cleanup(); assert.equal(stage.timers.size, 0);
});

test('camera pointer capture, keyboard and zoom clamp to supported ranges and reset absolutely', () => {
  const stage = stageSetup(); const target = { setPointerCapture(id) { this.captured = id; }, hasPointerCapture: () => true, releasePointerCapture(id) { this.released = id; } };
  stage.pointerDown({ button: 0, pointerId: 7, clientX: 10, clientY: 10, currentTarget: target });
  stage.pointerMove({ pointerId: 7, clientX: 30, clientY: 30 }); assert.equal(target.captured, 7); assert.equal(stage.camera.value.azimuth, 127);
  stage.pointerUp({ pointerId: 7, currentTarget: target }); assert.equal(target.released, 7);
  stage.updateCamera({ azimuth: 1000, elevation: 1000, distance: 100 }); assert.deepEqual({ ...stage.camera.value }, { azimuth: 360, elevation: -5, distance: 8 });
  let prevented = false; stage.keydown({ key: 'Home', preventDefault() { prevented = true; } }); assert.equal(prevented, true);
  assert.deepEqual({ ...stage.camera.value }, { azimuth: 135, elevation: -18, distance: 3 });
  stage.keydown({ key: '+', preventDefault() {} }); assert.ok(stage.camera.value.distance < 3); stage.wheel({ deltaY: 100 }); assert.ok(stage.camera.value.distance > 2.8);
  stage.updateCamera({ elevation: -1000, distance: -10 }); assert.equal(stage.camera.value.elevation, -85); assert.equal(stage.camera.value.distance, 1); stage.cleanup();
});

test('close is independent of initializing, connecting, disconnected, paused, fallen, finalizing and pending controls', async () => {
  for (const mode of ['initializing', 'connecting', 'disconnected', 'paused', 'fallen', 'finalizing', 'pending']) {
    let finish;
    const s = setup(async path => path.endsWith('/close') ? new Promise(resolve => { finish = resolve; }) : {});
    const ws = s.connect();
    if (mode === 'connecting') ws.readyState = 0;
    else if (mode === 'disconnected') ws.close();
    else if (mode === 'pending') s.command('task', 'move-ego-0-0.3');
    else ws.message({ type: 'state', state: snapshot(2, { status: mode, ...(mode === 'initializing' ? { body_poses: [] } : {}) }) });
    const first = s.closeEnvironment(); assert.equal(s.closeEnvironment(), first, mode);
    assert.equal(s.closing.value, true); assert.equal(s.canControl.value, false); assert.equal(s.closeNotice.value, '');
    assert.equal(s.job.value.status, 'running', 'HTTP request alone is not release confirmation');
    finish({ id: 'job_live', status: 'cancelled' }); assert.equal(await first, true);
    assert.equal(s.job.value.status, 'cancelled'); assert.match(s.closeNotice.value, /已关闭/);
    assert.equal(s.timers.size, 0); const count = s.sockets.length; s.reconnect(); await s.pollJob(); s.heartbeat();
    assert.equal(s.sockets.length, count); assert.equal(s.timers.size, 0); s.cleanup();
  }
});

test('SPA route leave awaits confirmed release and a failure blocks navigation until retry succeeds', async () => {
  let finish, reject = true;
  const s = setup(async path => {
    if (path.endsWith('/close')) { if (reject) throw Error('container removal failed'); return new Promise(resolve => { finish = resolve; }); }
    return {};
  });
  s.connect(); assert.equal(await s.leave(), false); assert.match(s.closeError.value, /container removal failed/);
  assert.equal(s.job.value.status, 'running'); assert.equal(s.connected.value, false);
  reject = false; let settled = false; const leaving = s.leave().then(result => { settled = true; return result; });
  await flush(); assert.equal(settled, false); assert.equal(s.closing.value, true);
  finish({ id: 'job_live', status: 'cancelled' }); assert.equal(await leaving, true); assert.equal(s.closeError.value, '');
  assert.equal(s.fallbacks.length, 0); s.cleanup(); assert.equal(s.fallbacks.length, 0, 'confirmed closure needs no unload fallback');
});

test('closing during an in-flight create uses its key immediately and does not wait for a lost create response', async () => {
  let created, released;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') return new Promise(resolve => { created = resolve; });
    if (path.endsWith('/close-pending')) return new Promise(resolve => { released = resolve; });
    return path.endsWith('/close-pending') ? { closed: true, job: null } : { items: [] };
  });
  await s.load(); await s.check(); const starting = s.start(); await flush();
  const createCall = s.calls.find(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST');
  assert.equal(createCall[1].signal, undefined, 'a browser abort cannot lose a committed Job ID');
  let left = false; const leaving = s.leave().then(value => { left = value; return value; });
  await flush(); assert.equal(left, false); assert.equal(s.calls.filter(([path]) => path.endsWith('/close-pending')).length, 1);
  released({ closed: true, job: { id: 'job_created', status: 'cancelled' } });
  assert.equal(await leaving, true); assert.equal(s.job.value.id, 'job_created'); assert.equal(s.job.value.status, 'cancelled');
  created({ id: 'job_created', status: 'queued' }); await starting;
  assert.equal(s.job.value.status, 'cancelled', 'late create response cannot overwrite confirmed closure');
  assert.equal(s.sockets.length, 0); s.cleanup();
});

test('a failed create without a known Job does not block leaving, and leaving during preview never creates one', async () => {
  let rejectCreate, finishPreview, phase = 'create';
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return phase === 'preview' ? new Promise(resolve => { finishPreview = resolve; }) : { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') return new Promise((_resolve, reject) => { rejectCreate = reject; });
    if (path.endsWith('/close-pending')) return { closed: true, job: null };
    return { items: [] };
  });
  await s.load(); await s.check(); const creating = s.start(); await flush(); const leaving = s.leave();
  rejectCreate(Error('creation failed')); await creating; assert.equal(await leaving, true); assert.equal(s.calls.filter(([path]) => path.endsWith('/close')).length, 0); s.cleanup();
  phase = 'preview';
  const p = setup(async path => path.endsWith('/capabilities') ? caps : path.endsWith('/preview') ? new Promise(resolve => { finishPreview = resolve; }) : { items: [] });
  await p.load(); const starting = p.start(); await flush(); assert.equal(await p.leave(), true);
  finishPreview({ plan: { hash: 'plan' } }); await starting;
  assert.equal(p.calls.some(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST'), false); p.cleanup();
});

test('pagehide and unmount close queued or unready environments with authenticated keepalive despite socket errors', async () => {
  const s = setup(); s.mount(); await flush(); const ws = s.connect();
  ws.fail = true; ws.failClose = true; s.cameraChanged({ azimuth: 140, elevation: -20, distance: 3 });
  s.event('pagehide'); assert.equal(s.connected.value, false); assert.equal(s.timers.size, 0);
  assert.equal(s.fallbacks.length, 1); const [path, init] = s.fallbacks[0];
  assert.equal(path, '/api/v1/ufo-native/live/sessions/job_live/close'); assert.equal(init.method, 'POST');
  assert.equal(init.credentials, 'include'); assert.equal(init.keepalive, true); assert.equal(init.signal, undefined);
  assert.ok(s.resets > 0); s.cleanup(); assert.equal(s.listeners.size, 0); assert.equal(s.timers.size, 0); await flush();
  const queued = setup(); queued.job.value = { id: 'job_queued', status: 'queued' }; queued.cleanup();
  assert.equal(queued.fallbacks[0][0], '/api/v1/ufo-native/live/sessions/job_queued/close');
});

test('an unmounted pending create still closes its returned Job without opening a WebSocket', async () => {
  let created;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') return new Promise(resolve => { created = resolve; });
    return { items: [] };
  });
  await s.load(); await s.check(); const starting = s.start(); await flush(); s.cleanup();
  created({ id: 'job_late', status: 'queued' }); await starting; await flush();
  assert.ok(s.fallbacks.some(([path]) => path.endsWith('/sessions/job_late/close')));
  assert.equal(s.sockets.length, 0); assert.equal(s.timers.size, 0);
});

test('BFCache restore refreshes real status and the owned list without restarting a closed live stream', async () => {
  const s = setup(async path => path.endsWith('/capabilities') ? caps : path.endsWith('/sessions/job_live') ? { id: 'job_live', status: 'cancelled' } : { items: [] });
  s.mount(); await flush(); s.connect(); s.event('pagehide'); await flush();
  await s.event('pageshow', { persisted: true });
  assert.equal(s.job.value.status, 'cancelled'); assert.equal(s.sockets.length, 1); assert.equal(s.connected.value, false);
  assert.equal([...s.timers.values()].some(timer => [500, 2000, 3000].includes(timer.ms)), false);
  assert.equal(s.calls.filter(([path]) => path.endsWith('/sessions')).length, 2); s.cleanup();
});

test('owned active environments load once, exclude the current Job, and individual closure supports retry', async () => {
  let fail = true;
  const items = [{ id: 'job_current', status: 'queued', model_id: 'model' }, { id: 'job_old', status: 'running', model_id: 'model' }];
  const s = setup(async path => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/sessions')) return { items };
    if (path.endsWith('/close')) { if (fail) throw Error('try again'); return { id: 'job_old', status: 'cancelled' }; }
    return {};
  });
  await s.load(); s.job.value = items[0]; assert.equal(s.otherSessions.value.length, 1); assert.equal(s.otherSessions.value[0].id, 'job_old');
  await s.closeOtherSession(items[1]); assert.equal(s.activeSessions.value.length, 2);
  fail = false; await s.closeOtherSession(items[1]); assert.equal(s.activeSessions.value.length, 1);
  assert.equal(s.job.value.id, 'job_current'); assert.equal(s.calls.filter(([path]) => path.endsWith('/sessions')).length, 1); s.cleanup();
});

test('old metadata responses cannot replace or reopen another session after a route query switch', async () => {
  let oldResponse;
  const s = setup(async path => {
    if (path.endsWith('/sessions/job_A')) return new Promise(resolve => { oldResponse = resolve; });
    if (path.endsWith('/sessions/job_A/close')) return { id: 'job_A', status: 'cancelled' };
    if (path.endsWith('/sessions/job_B')) return { id: 'job_B', status: 'running', ...identity, run_id: 'run_B' };
    return {};
  });
  s.job.value = { id: 'job_A', status: 'running', ...identity, run_id: 'run_A' };
  const polling = s.pollJob(); await flush(); assert.equal(await s.update({ query: { session_job: 'job_B' } }), true);
  assert.equal(s.job.value.id, 'job_B'); oldResponse({ id: 'job_A', status: 'running', ...identity, run_id: 'run_A' }); await polling;
  assert.equal(s.job.value.id, 'job_B'); assert.equal(s.sockets.length, 0);
  await s.tick(0); assert.equal(s.job.value.run_id, 'run_B'); assert.equal(s.sockets.length, 1);
  assert.match(s.sockets[0].url, /run_B\/stream$/); s.cleanup();
});

test('all terminal Job statuses remain idempotently closable and malformed success is never a release confirmation', async () => {
  for (const status of ['succeeded', 'failed', 'timeout', 'cancelled', 'infra_failed']) {
    const s = setup(async () => ({ id: 'job_terminal', status })); s.job.value = { id: 'job_terminal', status };
    assert.equal(await s.closeEnvironment(), true); assert.equal(await s.closeEnvironment(), true);
    assert.equal(s.calls.length, 1); s.cleanup();
  }
  const s = setup(async () => ({ id: 'job_live', status: 'running' })); s.connect();
  assert.equal(await s.closeEnvironment(), false); assert.equal(s.closeNotice.value, ''); assert.match(s.closeError.value, /尚未确认/); s.cleanup();
});

test('pagehide cancels the idempotency key immediately while a create response is still unresolved', async () => {
  let created;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') return new Promise(resolve => { created = resolve; });
    return { items: [] };
  });
  s.mount(); await flush(); await s.check(); const key = s.preview.value.key;
  const starting = s.start(); await flush(); s.event('pagehide');
  const pendingClose = s.fallbacks.find(([path]) => path.endsWith('/sessions/close-pending'));
  assert.ok(pendingClose, 'server must receive the key without waiting for JavaScript after navigation');
  assert.deepEqual(JSON.parse(pendingClose[1].body), { idempotency_key: key });
  assert.equal(pendingClose[1].headers['Content-Type'], 'application/json'); assert.equal(pendingClose[1].keepalive, true); assert.equal(pendingClose[1].credentials, 'include');
  assert.equal(s.job.value, null); s.cleanup();
  created({ id: 'job_late', status: 'queued' }); await starting; await flush(); assert.equal(s.sockets.length, 0);
});

test('an unknown create outcome must close its key before SPA navigation is allowed', async () => {
  let rejectCreate, failClose = true;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') return new Promise((_resolve, reject) => { rejectCreate = reject; });
    if (path.endsWith('/close-pending')) { if (failClose) throw Error('pending cleanup failed'); return { closed: true, job: null }; }
    return { items: [] };
  });
  await s.load(); await s.check(); const key = s.preview.value.key, creating = s.start(); await flush();
  const leaving = s.leave(); rejectCreate(Error('connection lost after submit')); await creating;
  assert.equal(await leaving, false); assert.match(s.closeError.value, /pending cleanup failed/);
  failClose = false; assert.equal(await s.leave(), true);
  const writes = s.calls.filter(([path]) => path.endsWith('/close-pending')); assert.equal(writes.length, 2);
  assert.deepEqual(JSON.parse(writes[1][1].body), { idempotency_key: key }); s.cleanup();
});

test('confirmed pending closure clears the busy form and an old response cannot replace a new environment', async () => {
  let oldResponse, creates = 0;
  const s = setup(async (path, init) => {
    if (path.endsWith('/capabilities')) return caps;
    if (path.endsWith('/preview')) return { plan: { hash: 'plan' } };
    if (path.endsWith('/sessions') && init?.method === 'POST') {
      if (++creates === 1) return new Promise(resolve => { oldResponse = resolve; });
      return { id: 'job_new', status: 'queued' };
    }
    if (path.endsWith('/close-pending')) return { closed: true, job: null };
    if (path.endsWith('/sessions/job_new')) return { id: 'job_new', status: 'queued' };
    return { items: [] };
  });
  await s.load(); await s.check(); const first = s.start(); await flush();
  assert.equal(await s.closeEnvironment(), true); assert.equal(s.busy.value, false);
  await s.start(); await flush(); assert.equal(s.job.value.id, 'job_new');
  const writes = s.calls.filter(([path, init]) => path.endsWith('/sessions') && init?.method === 'POST');
  assert.notEqual(writes[0][1].headers['Idempotency-Key'], writes[1][1].headers['Idempotency-Key']);
  oldResponse({ id: 'job_old', status: 'queued' }); await first; await flush();
  assert.equal(s.job.value.id, 'job_new'); assert.equal(s.busy.value, false); assert.equal(s.sockets.length, 0); s.cleanup();
});
