import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

// Exercise the component's actual setup logic without a browser or extra test dependency.
function setup(file, names, api) {
  const source = readFileSync(new URL(`../src/views/${file}.vue`, import.meta.url), 'utf8').match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const watchers = [];
  const route = { params: { id: 'first' } };
  const context = vm.createContext({ ref, computed, api, AbortController, console,
    onMounted() {}, onUnmounted() {}, watch(...args) { watchers.push(args); },
    useRouter: () => ({ push() {} }), useRoute: () => route,
    clearTimeout() {}, window: { setTimeout() { return 1; } },
  });
  const code = ts.transpile(source.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.subject = { ${names.join(',')} };`, context);
  return { ...context.subject, watchers, route };
}

test('template required inputs and nullable defaults are used in the submitted payload', () => {
  const s = setup('TrainNew', ['form', 'templates', 'payload', 'missingInput'], async () => ({}));
  s.templates.value = [{ id: 'sft', requires_dataset: true, requires_base_model: true }];
  Object.assign(s.form.value, { name: ' test ', template_id: 'sft', embodiment_id: 'franka' });
  assert.equal(s.missingInput.value, true);
  s.form.value.dataset_ids = ['dataset'];
  s.form.value.base_model_id = 'base';
  assert.equal(s.missingInput.value, false);
  const p = s.payload();
  for (const key of ['max_epochs', 'max_steps', 'total_num_envs', 'micro_batch_size', 'global_batch_size', 'lr']) assert.equal(p[key], null);
  assert.equal(p.name, 'test');
  assert.equal('auto_register_latest' in p, false);
  s.form.value.max_steps = '10';
  assert.equal(s.payload().max_steps, 10);
});

test('PPO recipes expose micro and global batch inputs', () => {
  const source = readFileSync(new URL('../src/views/TrainNew.vue', import.meta.url), 'utf8');
  assert.match(source, /v-if="selectedTemplate\?\.kind === 'sft' \|\| selectedTemplate\?\.kind === 'ppo'"/);
});

test('online PPO step limit fills its matching RLinf epoch limit', () => {
  const s = setup('TrainNew', ['form', 'templates', 'payload'], async () => ({}));
  s.templates.value = [{ id: 'rpo_v2_wbc_ppo', kind: 'ppo' }];
  s.form.value.template_id = 'rpo_v2_wbc_ppo';
  s.form.value.max_steps = '100';
  assert.equal(s.payload().max_epochs, 100);
  assert.equal(s.payload().max_steps, 100);
});

test('a visible but blocked RoboTwin recipe cannot be submitted', () => {
  const s = setup('TrainNew', ['form', 'templates', 'missingInput', 'selectedTemplate'], async () => ({}));
  s.templates.value = [{ id: 'robotwin_g0_sft', ready: false, blocked_reason: 'worker missing',
    requires_dataset: true, requires_base_model: true }];
  Object.assign(s.form.value, { name: 'g0', template_id: 'robotwin_g0_sft',
    embodiment_id: 'emb_robotwin_aloha_agilex', dataset_ids: ['data'], base_model_id: 'base' });
  assert.equal(s.selectedTemplate.value.blocked_reason, 'worker missing');
  assert.equal(s.missingInput.value, true);
});

test('robot selection exposes only datasets from its environments and clears stale data', () => {
  const s = setup('TrainNew', ['form', 'templates', 'embodiments', 'datasets', 'compatibleDatasets'], async () => ({}));
  s.templates.value = [{ id: 'sft' }];
  s.embodiments.value = [
    { id: 'franka', envs: [{ id: 'ee_franka' }] },
    { id: 'so101', envs: [{ id: 'ee_so101' }] },
  ];
  s.datasets.value = [
    { id: 'libero', embodiment_env_id: 'ee_franka' },
    { id: 'arm', embodiment_env_id: 'ee_so101' },
    { id: 'unknown', embodiment_env_id: null },
  ];
  Object.assign(s.form.value, { template_id: 'sft', embodiment_id: 'franka', dataset_ids: ['libero'] });
  assert.deepEqual(Array.from(s.compatibleDatasets.value, d => d.id), ['libero']);
  s.form.value.embodiment_id = 'so101';
  s.watchers[1][1]();
  assert.deepEqual(Array.from(s.compatibleDatasets.value, d => d.id), ['arm']);
  assert.deepEqual(Array.from(s.form.value.dataset_ids), []);
  s.form.value.dataset_ids = ['arm'];
  s.watchers[0][1]();
  assert.deepEqual(Array.from(s.form.value.dataset_ids), []);
});

test('base-model choices match the selected embodiment and stale selection is cleared', () => {
  const s = setup('TrainNew', ['form', 'weights', 'compatibleWeights'], async () => ({}));
  s.weights.value = [
    { id: 'libero', embodiment_id: 'emb_libero_franka' },
    { id: 'robotwin', embodiment_id: 'emb_robotwin_aloha_agilex' },
  ];
  Object.assign(s.form.value, { embodiment_id: 'emb_robotwin_aloha_agilex', base_model_id: 'libero' });
  assert.deepEqual(Array.from(s.compatibleWeights.value, w => w.id), ['robotwin']);
  s.watchers[1][1]();
  assert.equal(s.form.value.base_model_id, '');
});

test('a stale preview response cannot replace newer form state', async () => {
  let resolve;
  const s = setup('TrainNew', ['preview', 'refreshPreview'], () => new Promise(r => { resolve = r; }));
  const pending = s.refreshPreview();
  s.watchers.find(w => typeof w[0] !== 'function')[1]();
  resolve({ image: 'stale' });
  await pending;
  assert.equal(s.preview.value, null);
});

test('pipeline PPO epochs are independent from the SFT payload and pending prevents duplicates', async () => {
  const requests = [];
  let resolve;
  const s = setup('TrainNew', ['form', 'templates', 'payload', 'createPipeline'], (path, options) => {
    requests.push({ path, body: JSON.parse(options.body) });
    return new Promise(r => { resolve = r; });
  });
  s.templates.value = [{ id: 'libero_pi05_sft', requires_dataset: true, requires_base_model: true }];
  Object.assign(s.form.value, { name: 'pipeline', template_id: 'libero_pi05_sft',
    embodiment_id: 'franka', dataset_ids: ['dataset'], base_model_id: 'base', max_steps: '2', max_epochs: '7' });
  assert.equal(s.form.value.ppo_max_epochs, '1');
  s.form.value.ppo_max_epochs = '2';
  assert.equal('ppo_max_epochs' in s.payload(), false);
  const pending = s.createPipeline();
  await s.createPipeline();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].path, '/train/libero-pipeline');
  assert.equal(requests[0].body.ppo_max_epochs, 2);
  assert.equal(requests[0].body.max_epochs, 7);
  assert.equal(requests[0].body.max_steps, 2);
  resolve({ id: 'pipeline' });
  await pending;
});

test('changing job routes ignores stale loads; metric names keep their actual semantics', async () => {
  const pending = new Map();
  const s = setup('TrainJobDetail', ['load', 'job', 'metrics', 'metricNames'], path => {
    if (path.endsWith('/checkpoints')) return Promise.resolve({ items: [] });
    if (path.endsWith('/resume-candidate')) return Promise.resolve({ available: false });
    if (path.includes('/metrics')) return Promise.resolve({ series: {} });
    if (path.includes('/logs')) return Promise.resolve({ lines: [] });
    return new Promise(resolve => pending.set(path, resolve));
  });
  const first = s.load();
  s.route.params.id = 'second';
  const second = s.load();
  pending.get('/train/second')({ id: 'second' });
  await second;
  pending.get('/train/first')({ id: 'first' });
  await first;
  assert.equal(s.job.value.id, 'second');
  s.metrics.value = { series: { 'train/loss': [{ step: 1, value: 0.2 }] } };
  assert.deepEqual(Array.from(s.metricNames.value), ['train/loss']);
});
