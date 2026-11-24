import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

function setup(file, names, api = async () => ({})) {
  const source = readFileSync(new URL(`../src/views/${file}.vue`, import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const watchers = [], pushes = [], unmounts = [];
  const route = { params: { id: 'job_depth' }, query: {} };
  const context = vm.createContext({ ref, computed, api, AbortController,
    watch(...args) { watchers.push(args); }, onMounted() {}, onUnmounted(fn) { unmounts.push(fn); }, clearTimeout() {},
    window: { setTimeout() {} }, useRoute: () => route,
    useRouter: () => ({ push(path) { pushes.push(path); } }),
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + `\nglobalThis.subject = { ${names.join(',')} };`, context);
  return { ...context.subject, watchers, pushes, route, unmounts };
}
const template = 'rpo_pbfm_fb_depth_native', env = 'ee_rpo_v2_pbfm_depth_23dof_v1';
const capability = { mode: 'training_validation', gpus: 4, num_envs_per_rank: 2048,
  buffer_size: 131072, seed: 42, target_updates: 112, stages: [112, 368, 512], timeout_seconds: 21600 };
function configure(s) {
  s.templates.value = [{ id: template, kind: 'native_rl', ready: false,
    controlled_training: capability, embodiment_env_id: env, requires_dataset: true }];
  s.environments.value = [{ id: 'profile', status: 'draft', usage: 'train', template_id: template, embodiment_env_id: env }];
  Object.assign(s.form.value, { name: ' depth ', template_id: template, environment_id: 'profile',
    embodiment_id: 'emb_roboparty_rpo', dataset_ids: ['ds_depth'] });
}

test('only the explicit four-card capability allows a draft profile and frozen cold-start request', () => {
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'missingInput', 'payload']);
  configure(s);
  assert.equal(s.missingInput.value, false);
  assert.deepEqual(JSON.parse(JSON.stringify(s.payload())), { name: 'depth', dataset_id: 'ds_depth',
    environment_id: 'profile', mode: 'training_validation', gpus: 4, num_envs_per_rank: 2048,
    target_updates: 112, timeout_seconds: 21600 });
  s.templates.value[0].controlled_training = undefined;
  assert.equal(s.missingInput.value, true);
});

test('protected preview and submit preserve the hash and retry key and navigate using original job_id', async () => {
  const calls = [], hash = `sha256:${'a'.repeat(64)}`;
  let failures = 1;
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'refreshPreview', 'preview', 'submit'], async (path, options) => {
    calls.push({ path, body: JSON.parse(options.body), key: options.headers?.['Idempotency-Key'] });
    if (path.endsWith('/preview')) return { preview_hash: hash };
    if (failures--) throw Error('network interrupted');
    return { job_id: 'job_created' };
  });
  configure(s);
  await s.refreshPreview();
  assert.equal(calls[0].path, '/pbfm/runtime-probe/preview');
  await s.submit(); await s.submit();
  assert.equal(calls[1].path, '/pbfm/runtime-probe');
  assert.equal(calls[1].body.preview_hash, hash);
  assert.ok(calls[1].key); assert.equal(calls[1].key, calls[2].key);
  assert.deepEqual(s.pushes, ['/train/job_created']);
});

test('changing the form invalidates an older in-flight preview', async () => {
  let finish;
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'refreshPreview', 'preview'],
    () => new Promise(resolve => { finish = resolve; }));
  configure(s);
  const pending = s.refreshPreview();
  s.form.value.name = 'changed'; s.watchers[2][1]();
  finish({ preview_hash: 'stale' }); await pending;
  assert.equal(s.preview.value, null);
});

test('progress counts actual joint updates, with the resume offset and unknown state preserved', () => {
  const s = setup('TrainJobDetail', ['job', 'metrics', 'nativeProgress']);
  s.job.value = { template_id: template, kind: 'native_rl', params: { native_train_plan: { start_updates: 368, target_updates: 512 } }, metrics_summary: {} };
  assert.equal(s.nativeProgress.value.actual, null);
  s.metrics.value = { series: { 'distributed/optimizer_steps': [{ step: 400, value: 400 }] } };
  assert.equal(s.nativeProgress.value.actual, 400);
  assert.equal(s.nativeProgress.value.completed, 32);
  assert.equal(s.nativeProgress.value.total, 144);
  s.job.value.metrics_summary.optimizer_steps = 512;
  assert.equal(s.nativeProgress.value.completed, 144);
});

test('single P-BFM exploration shows 512→10000 with exactly9488 new updates', () => {
  const s = setup('TrainJobDetail', ['job', 'nativeProgress', 'isPbfmExploration']);
  s.job.value = { template_id: template, kind: 'native_rl', record_kind: 'pbfm_validation',
    training_mode: 'training_exploration', params: { native_train_plan: { start_updates: 512, target_updates: 10000 } },
    metrics_summary: { optimizer_steps: 10000 } };
  assert.equal(s.isPbfmExploration.value, true);
  assert.equal(s.nativeProgress.value.actual, 10000);
  assert.equal(s.nativeProgress.value.total, 9488);
  assert.equal(s.nativeProgress.value.completed, 9488);
});

test('P-BFM limits default charts while keeping every metric selectable', () => {
  const s = setup('TrainJobDetail', ['job', 'metrics', 'metricNames', 'visibleMetricNames', 'selectedMetric']);
  s.job.value = { template_id: template };
  s.metrics.value = { series: Object.fromEntries(Array.from({ length: 260 }, (_, i) => [`train/loss_${i}`, [{ step: 1, value: i }]])) };
  assert.equal(s.metricNames.value.length, 260);
  assert.ok(s.visibleMetricNames.value.length > 0 && s.visibleMetricNames.value.length <= 14);
  s.selectedMetric.value = 'train/loss_259';
  assert.deepEqual(Array.from(s.visibleMetricNames.value), ['train/loss_259']);
});

test('100000 exploration displays its own protocol and real90000-update progress', () => {
  const s = setup('TrainJobDetail', ['job', 'nativeProgress', 'isPbfmExploration', 'pbfmExplorationProtocol']);
  s.job.value = { template_id: template, training_mode: 'training_long_exploration',
    params: { native_train_plan: { start_updates: 10000, target_updates: 100000 } },
    metrics_summary: { optimizer_steps: 19984 } };
  assert.equal(s.isPbfmExploration.value, true);
  assert.match(s.pbfmExplorationProtocol.value, /10000.*100000.*90000/);
  assert.match(s.pbfmExplorationProtocol.value, /30小时/);
  assert.equal(s.nativeProgress.value.completed, 9984);
  assert.equal(s.nativeProgress.value.total, 90000);
  s.job.value.training_mode = 'training_exploration';
  assert.match(s.pbfmExplorationProtocol.value, /512.*10000.*9488/);
});

test('continuation uses its reviewed source and report through the protected submit endpoint', async () => {
  const calls = [], hash = `sha256:${'b'.repeat(64)}`;
  const s = setup('TrainJobDetail', ['job', 'resumeCandidate', 'openResume', 'resumeBody', 'previewResume', 'submitResume'], async (path, options) => {
    calls.push({ path, body: JSON.parse(options.body), key: options.headers?.['Idempotency-Key'] });
    return path.endsWith('/preview') ? { preview_hash: hash } : { job_id: 'job_next' };
  });
  s.job.value = { id: 'job_depth', name: 'depth', kind: 'native_rl', template_id: template, record_kind: 'pbfm_validation' };
  s.resumeCandidate.value = { available: true, step: 112, termination: 'target_updates', suggested_target: 368,
    request: { dataset_id: 'ds_depth', environment_id: 'profile', mode: 'training_validation', gpus: 4,
      num_envs_per_rank: 2048, target_updates: 368, resume_job_id: 'job_depth', validation_report_id: 'ev_depth', timeout_seconds: 21600 } };
  s.openResume(); await s.previewResume(); await s.submitResume();
  assert.equal(calls[0].path, '/pbfm/runtime-probe/preview');
  assert.equal(calls[1].path, '/pbfm/runtime-probe');
  assert.equal(calls[1].body.resume_job_id, 'job_depth');
  assert.equal(calls[1].body.validation_report_id, 'ev_depth');
  assert.equal(calls[1].body.target_updates, 368);
  assert.equal(calls[1].body.preview_hash, hash); assert.ok(calls[1].key);
  assert.deepEqual(s.pushes, ['/train/job_next']);
});

function resumeFixture(s) {
  s.job.value = { id: 'job_depth', name: 'depth', kind: 'native_rl', template_id: template, record_kind: 'pbfm_validation' };
  s.resumeCandidate.value = { available: true, step: 112, suggested_target: 368,
    request: { mode: 'training_validation', target_updates: 368 } };
  s.openResume();
}
for (const fail of [false, true]) test(`late resume ${fail ? 'failure' : 'success'} cannot affect a different route`, async () => {
  let resolve, reject;
  const s = setup('TrainJobDetail', ['job', 'resumeCandidate', 'openResume', 'resumePreview', 'submitResume', 'err'],
    () => new Promise((yes, no) => { resolve = yes; reject = no; }));
  resumeFixture(s); s.resumePreview.value = { preview_hash: `sha256:${'c'.repeat(64)}` };
  const pending = s.submitResume();
  s.route.params.id = 'job_other';
  s.job.value = { id: 'job_other' }; s.err.value = 'current page';
  if (fail) reject(Error('old submit failed')); else resolve({ job_id: 'job_late' });
  await pending;
  assert.deepEqual(s.pushes, []); assert.equal(s.err.value, 'current page');
});

test('unmount aborts a resume preview and ignores its late error', async () => {
  let reject, signal;
  const s = setup('TrainJobDetail', ['job', 'resumeCandidate', 'openResume', 'previewResume', 'err'],
    (_path, options) => { signal = options.signal; return new Promise((_yes, no) => { reject = no; }); });
  resumeFixture(s);
  const pending = s.previewResume();
  s.unmounts.forEach(fn => fn());
  reject(Error('old preview failed')); await pending;
  assert.equal(s.err.value, ''); assert.equal(signal.aborted, true);
});

test('pending continuation submit cannot create a duplicate request', async () => {
  const resolvers = [];
  const s = setup('TrainJobDetail', ['job', 'resumeCandidate', 'openResume', 'resumePreview', 'submitResume'],
    () => new Promise(yes => { resolvers.push(yes); }));
  resumeFixture(s); s.resumePreview.value = { preview_hash: `sha256:${'c'.repeat(64)}` };
  const pending = s.submitResume(), duplicate = s.submitResume();
  resolvers.forEach(resolve => resolve({ job_id: 'job_next' }));
  await Promise.all([pending, duplicate]);
  assert.equal(resolvers.length, 1);
});
