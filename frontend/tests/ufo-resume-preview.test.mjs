import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { computed, ref, watch } from 'vue';

function setup(api) {
  const calls = [], navigations = [], stops = [];
  const source = readFileSync(new URL('../src/views/TrainJobDetail.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, AbortController,
    watch(source, callback, options) {
      // Run the component's actual form watcher without starting route polling.
      if (Array.isArray(source)) stops.push(watch(source, callback, options));
    },
    useRoute: () => ({ params: { id: 'trn_native' } }),
    useRouter: () => ({ push(path) { navigations.push(path); } }),
    onUnmounted() {}, clearTimeout() {}, window: { setTimeout() {} },
    api: async (path, options) => { calls.push([path, options]); return api(path, options); },
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + '\nglobalThis.subject = { job, resumeCandidate, resumeName, resumeModelName, resumeTarget, resumePreview, resumeKey, previewResume, submitResume, err };', context);
  const s = context.subject;
  s.job.value = { id: 'trn_native', kind: 'native_rl' };
  s.resumeCandidate.value = { termination: 'target_updates', step: 112 };
  s.resumeName.value = 'continued';
  s.resumeModelName.value = 'model-512';
  s.resumeTarget.value = 512;
  s.resumeKey.value = 'fixed-resume-key';
  return { ...s, calls, navigations, stop() { stops.forEach(stop => stop()); } };
}

test('each editable resume input synchronously invalidates the reviewed preview', () => {
  const s = setup(async () => ({}));
  for (const [field, value] of [['resumeName', 'renamed'], ['resumeModelName', 'other'], ['resumeTarget', 1008]]) {
    s.resumePreview.value = { preview_hash: `sha256:${'a'.repeat(64)}` };
    s[field].value = value;
    assert.equal(s.resumePreview.value, null);
  }
  s.stop();
});

test('an old asynchronous preview cannot enable submission after the target changes', async () => {
  let resolve;
  const s = setup(() => new Promise(done => { resolve = done; }));
  const pending = s.previewResume();
  s.resumeTarget.value = 1008;
  resolve({ preview_hash: `sha256:${'a'.repeat(64)}` });
  await pending;
  assert.equal(s.resumePreview.value, null);
  await s.submitResume();
  assert.equal(s.calls.length, 1);
  assert.match(s.err.value, /重新预览/);
  s.stop();
});

test('native resume sends the matching preview hash and preserves the idempotency key', async () => {
  const hash = `sha256:${'b'.repeat(64)}`;
  const s = setup(async path => path.endsWith('/preview') ? { preview_hash: hash } : { id: 'trn_child' });
  await s.previewResume();
  await s.submitResume();
  assert.deepEqual(JSON.parse(s.calls[0][1].body), {
    name: 'continued', model_name: 'model-512', target_updates: 512,
  });
  assert.deepEqual(JSON.parse(s.calls[1][1].body), {
    name: 'continued', model_name: 'model-512', target_updates: 512, preview_hash: hash,
  });
  assert.equal(s.calls[1][1].headers['Idempotency-Key'], 'fixed-resume-key');
  assert.deepEqual(s.navigations, ['/train/trn_child']);
  s.stop();
});

test('native resume requires a hash while older stacks retain their preview contract', async () => {
  const s = setup(async () => ({ id: 'trn_legacy_child' }));
  s.resumePreview.value = { argv: ['legacy'] };
  await s.submitResume();
  assert.equal(s.calls.length, 0);
  s.job.value = { id: 'trn_sft', kind: 'sft' };
  s.resumeCandidate.value = { termination: 'max_steps', step: 100 };
  await s.submitResume();
  assert.deepEqual(JSON.parse(s.calls[0][1].body), {
    name: 'continued', model_name: 'model-512', max_steps: 512,
  });
  s.stop();
});
