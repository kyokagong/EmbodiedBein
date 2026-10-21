import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { computed, ref } from 'vue';

function setup(model, respond = async () => ({})) {
  const calls = [], events = [], timers = new Map(), watchers = [];
  let cleanup, nextTimer = 0;
  const source = readFileSync(new URL('../src/components/UfoNativePanel.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, AbortController,
    defineProps: () => ({ model }), defineEmits: () => event => events.push(event),
    onMounted() {}, onUnmounted(fn) { cleanup = fn; }, watch(...args) { watchers.push(args); },
    setTimeout(fn) { const id = ++nextTimer; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
    crypto: { randomUUID: () => 'preview-idempotency-key' },
    api: async (path, init) => { calls.push([path, init]); return respond(path, init); },
  });
  const code = ts.transpile(source.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.subject = { check, submit, refresh, preview, busy, error, sourceName, sourcePath, backend, profile, lastJob, status, promptSeed, promptSamples, promptInputValid, currentModel, trained, exportChildren };`, context);
  return { ...context.subject, calls, events, timers, watchers, cleanup: () => cleanup() };
}

test('native imports bind preview to inputs and reuse the idempotency key on retry', async () => {
  let failed = false, finished = false;
  const s = setup(undefined, async (path, init) => {
    if (path.endsWith('/preview')) return { plan: { hash: 'frozen-source' } };
    if (path.endsWith('/imports')) {
      if (!failed) { failed = true; throw new Error('temporary failure'); }
      return { id: 'job_import', status: 'queued' };
    }
    return { id: 'job_import', status: finished ? 'succeeded' : 'queued' };
  });
  s.sourceName.value = 'native model'; s.sourcePath.value = '/source/model';
  await s.check('imports');
  s.watchers[0][1]();
  await s.submit();
  assert.equal(s.calls.length, 1, 'changing inputs invalidates the checked plan');
  await s.check('imports');
  await s.submit();
  assert.match(s.error.value, /temporary failure/);
  await s.submit();
  const writes = s.calls.filter(([path]) => path.endsWith('/imports'));
  assert.equal(writes.length, 2);
  assert.equal(writes[0][1].headers['Idempotency-Key'], writes[1][1].headers['Idempotency-Key']);
  assert.deepEqual(JSON.parse(writes[1][1].body), { name: 'native model', path: '/source/model', preview_hash: 'frozen-source' });
  assert.equal(s.preview.value, null);
  finished = true;
  const [timerId, poll] = s.timers.entries().next().value;
  s.timers.delete(timerId); await poll();
  assert.deepEqual(s.events, ['changed'], 'completed imports refresh the parent model list');
  assert.equal(s.timers.size, 0);
  s.cleanup();
});

test('model simulation uses the selected frozen backend/profile and stops polling at completion', async () => {
  let complete = false;
  const s = setup({ id: 'mv_native', card: { schema_version: 'eb/rpo-ufo-native-export/v1' } }, async path => {
    if (path.endsWith('/preview')) return { plan: { hash: 'eval-plan' } };
    if (path.endsWith('/evals')) return { id: 'job_eval', status: 'queued' };
    return { jobs: [{ id: 'job_eval', status: complete ? 'succeeded' : 'running' }], reports: [] };
  });
  s.backend.value = 'mujoco'; s.profile.value = 'acceptance';
  await s.check('evals'); await s.submit();
  const write = s.calls.find(([path]) => path.endsWith('/evals'));
  assert.equal(write[0], '/ufo-native/models/mv_native/evals');
  assert.deepEqual(JSON.parse(write[1].body), { backend: 'mujoco', profile: 'acceptance', preview_hash: 'eval-plan' });
  assert.equal(s.timers.size, 1);
  complete = true;
  const [id, poll] = s.timers.entries().next().value;
  s.timers.delete(id); await poll();
  assert.equal(s.timers.size, 0, 'terminal jobs do not poll indefinitely');
  s.cleanup();
});

test('unmounted panels ignore late status responses and clear polling', async () => {
  let resolve;
  const s = setup({ id: 'mv_old' }, () => new Promise(done => { resolve = done; }));
  const pending = s.refresh();
  s.cleanup();
  resolve({ model: { id: 'mv_old' }, jobs: [{ id: 'job_old', status: 'running' }] });
  await pending;
  assert.equal(s.status.value, null);
  assert.equal(s.timers.size, 0);
});

test('active jobs stop automatic polling after the ten-minute budget', async () => {
  const s = setup({ id: 'mv_native' }, async () => ({ jobs: [{ id: 'job_long', status: 'running' }] }));
  await s.refresh();
  let polls = 0;
  while (s.timers.size && polls < 125) {
    const [id, poll] = s.timers.entries().next().value;
    s.timers.delete(id); await poll(); polls++;
  }
  assert.equal(polls, 120);
  assert.equal(s.timers.size, 0);
  s.cleanup();
});

test('trained UFO model derives task prompts only from a preview, preserving source readiness', async () => {
  const model = { id: 'mv_trained', train_page_id: 'trn_source', card: {
    schema_version: 'eb/rpo-ufo-native-trained-model/v1', readiness: 'prompt_required',
    training_provenance: 'platform', run_id: 'run_train', counts: { optimizer_steps: 112 },
  } };
  let failed = false, complete = false;
  const child = { id: 'mv_prompted', card: { schema_version: 'eb/rpo-ufo-native-model/v1' } };
  const s = setup(model, async path => {
    if (path.endsWith('/preview')) return { plan: { hash: 'prompt-plan' }, resources: { gpus: 0 } };
    if (path.endsWith('/prompt')) {
      if (!failed) { failed = true; throw new Error('temporary prompt failure'); }
      return { id: 'job_prompt', status: 'queued' };
    }
    return { model: { id: model.id, card: model.card },
      jobs: [{ id: 'job_prompt', status: complete ? 'succeeded' : 'running' }],
      exports: complete ? [child] : [] };
  });
  await s.submit();
  assert.equal(s.calls.length, 0, 'prompt submission requires a checked plan');
  assert.equal(s.trained.value, true);
  assert.equal(s.promptSeed.value, 71000);
  assert.equal(s.promptSamples.value, 16384);
  await s.check('prompt');
  assert.equal(s.calls[0][0], '/ufo-native/models/mv_trained/prompt/preview');
  assert.deepEqual(JSON.parse(s.calls[0][1].body), { seed: 71000, num_samples: 16384 });
  await s.submit();
  assert.match(s.error.value, /temporary prompt failure/);
  await s.submit();
  const writes = s.calls.filter(([path]) => path.endsWith('/prompt'));
  assert.equal(writes.length, 2);
  assert.equal(writes[0][1].headers['Idempotency-Key'], writes[1][1].headers['Idempotency-Key']);
  assert.deepEqual(JSON.parse(writes[1][1].body), {
    seed: 71000, num_samples: 16384, preview_hash: 'prompt-plan',
  });
  assert.equal(s.timers.size, 1);
  complete = true;
  const [id, poll] = s.timers.entries().next().value;
  s.timers.delete(id); await poll();
  assert.equal(s.timers.size, 0);
  assert.equal(s.status.value.exports[0].id, child.id);
  assert.equal(s.exportChildren.value.length, 0, 'a derived prompt package is not an ONNX export');
  assert.equal(s.currentModel.value.train_page_id, 'trn_source', 'lightweight status keeps training lineage');
  assert.equal(model.card.readiness, 'prompt_required');
  assert.equal(s.currentModel.value.card.readiness, 'prompt_required');
  s.cleanup();
});

test('prompt inputs invalidate stale previews and reject unsupported sampling budgets', async () => {
  let resolve;
  const s = setup({ id: 'mv_trained', card: { schema_version: 'eb/rpo-ufo-native-trained-model/v1' } },
    () => new Promise(done => { resolve = done; }));
  const pending = s.check('prompt');
  s.promptSeed.value = 71001;
  s.watchers[0][1]();
  resolve({ plan: { hash: 'old-seed' } });
  await pending;
  assert.equal(s.preview.value, null);
  s.promptSamples.value = 257;
  assert.equal(s.promptInputValid.value, false);
  await s.check('prompt');
  assert.equal(s.calls.length, 1, 'invalid inputs are not sent to the API');
  assert.match(s.error.value, /256/);
  s.cleanup();
});
