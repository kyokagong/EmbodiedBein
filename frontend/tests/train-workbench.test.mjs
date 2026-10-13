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
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'payload', 'missingInput'], async () => ({}));
  s.templates.value = [{ id: 'sft', requires_dataset: true, requires_base_model: true }];
  s.environments.value = [{ id: 'train_env', status: 'active', usage: 'train', template_id: 'sft' }];
  Object.assign(s.form.value, { name: ' test ', model_name: 'model', environment_id: 'train_env', template_id: 'sft', embodiment_id: 'franka' });
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

test('PPO termination inputs stay explicit in the payload; the server resolves its epoch limit', () => {
  const s = setup('TrainNew', ['form', 'templates', 'payload'], async () => ({}));
  s.templates.value = [{ id: 'rpo_v2_wbc_ppo', kind: 'ppo' }];
  s.form.value.template_id = 'rpo_v2_wbc_ppo';
  s.form.value.max_steps = '100';
  assert.equal(s.payload().max_epochs, null);
  assert.equal(s.payload().max_steps, 100);
  s.form.value.max_epochs = '20';
  assert.equal(s.payload().max_epochs, 20);
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
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'payload', 'createPipeline'], (path, options) => {
    requests.push({ path, body: JSON.parse(options.body) });
    return new Promise(r => { resolve = r; });
  });
  s.templates.value = [{ id: 'libero_pi05_sft', requires_dataset: true, requires_base_model: true }];
  s.environments.value = [{ id: 'train_env', status: 'active', usage: 'train', template_id: 'libero_pi05_sft' }];
  Object.assign(s.form.value, { name: 'pipeline', model_name: 'model', environment_id: 'train_env', template_id: 'libero_pi05_sft',
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

test('training losses precede operational timings without dropping raw metrics', () => {
  const s = setup('TrainJobDetail', ['metrics', 'metricNames'], async () => ({}));
  const series = {
    'env/num_trajectories': [{ step: 3, value: 0 }],
    'rollout/bfm/collected_transitions': [{ step: 3, value: 16 }],
    'time/step': [{ step: 3, value: 7.319 }],
    'train/bfm/actor_grad_norm': [{ step: 3, value: 0.3467 }],
    'train/bfm/actor_loss': [{ step: 3, value: -0.014269933104515076 }],
    'train/bfm/fb_loss': [{ step: 3, value: 229991.625 }],
    'train/bfm/update_count': [{ step: 3, value: 4 }],
  };
  s.metrics.value = { series };
  const names = Array.from(s.metricNames.value);
  assert.deepEqual(names.slice(0, 2), ['train/bfm/actor_loss', 'train/bfm/fb_loss']);
  assert.equal(names.at(-1), 'time/step');
  assert.deepEqual([...names].sort(), Object.keys(series).sort());
  assert.equal(Object.keys(series)[0], 'env/num_trajectories', 'display sorting must not mutate the receipt');
});

test('BFM summary uses measured latest values, preserving zero, negatives and missing data', () => {
  const s = setup('TrainJobDetail', ['job', 'metrics', 'isBfm', 'bfmSummary', 'singleSample'], async () => ({}));
  s.job.value = { template_id: 'rpo_v2_bfm_zero_pretrain', resume_step: 3 };
  assert.equal(s.isBfm.value, true);
  assert.equal(s.singleSample.value, false);
  assert.ok(s.bfmSummary.value.every(metric => metric.value === null));
  s.metrics.value = { series: {
    'train/bfm/actor_loss': [{ step: 3, value: -0.014269933104515076 }],
    'train/bfm/fb_loss': [{ step: 3, value: 229991.625 }],
    'train/bfm/update_count': [{ step: 3, value: 4 }],
    'rollout/bfm/collected_transitions': [{ step: 3, value: 0 }],
  } };
  const values = Object.fromEntries(s.bfmSummary.value.map(metric => [metric.name, metric.value]));
  assert.equal(values['train/bfm/fb_loss'], 229991.625);
  assert.equal(values['train/bfm/actor_loss'], -0.014269933104515076);
  assert.equal(values['train/bfm/update_count'], 4);
  assert.equal(values['rollout/bfm/collected_transitions'], 0);
  assert.equal(values['train/bfm/critic_loss'], null, 'absent loss must not be replaced with zero');
  assert.equal(s.singleSample.value, true, 'resume step must not be mistaken for four samples');
  s.metrics.value.series['train/bfm/actor_loss'].push({ step: 4, value: -0.02 });
  assert.equal(s.singleSample.value, false);
  assert.equal(s.bfmSummary.value.find(metric => metric.name === 'train/bfm/actor_loss').value, -0.02);
  s.job.value = { template_id: 'rpo_v2_flat_ppo' };
  assert.equal(s.isBfm.value, false);
});

test('UFO sends absolute joint-update targets and qualification identity', () => {
  const s = setup('TrainNew', ['form', 'templates', 'payload'], async () => ({}));
  s.templates.value = [{id:'ufo', execution:{provider_id:'ufo'}}];
  Object.assign(s.form.value, {template_id:'ufo', name:'test', model_name:'model', gpus:1,
    purpose:'qualification', target_learner_updates:'24'});
  const body = s.payload();
  assert.equal(body.target_learner_updates, 24);
  assert.equal(body.purpose, 'qualification');
  assert.equal(body.max_epochs, undefined);
  assert.equal(body.max_steps, undefined);
});

test('UFO four-card recipe submits only supported knobs after loading fixed defaults', () => {
  const s = setup('TrainNew', ['form', 'templates', 'payload'], async () => ({}));
  s.templates.value = [{id:'ufo_four', execution:{provider_id:'ufo', topology_id:'one_learner_three_samplers'},
    default_overrides:['env.train.total_num_envs=192', 'env.eval.total_num_envs=3', 'actor.seed=42']}];
  s.form.value.template_id = 'ufo_four';
  s.watchers[0][1]();
  assert.equal(s.form.value.gpus, 4);
  assert.equal(s.payload().seed, 42);
  for (const field of ['total_num_envs', 'eval_num_envs', 'lr', 'value_lr', 'extra_overrides'])
    assert.equal(field in s.payload(), false);
});

test('UFO evaluation suites and models remain selectable with required video', () => {
  const s = setup('EvalNew', ['suites', 'suiteId', 'source', 'profile', 'rpoCandidates', 'visibleSuites', 'compatibleRpoCandidates', 'body'], async () => ({}));
  s.suites.value = [{id:'ufo_validation', embodiment_env_id:'rpo_bfm', spec:{benchmark:'rpo_bfm', backend:'isaac', evaluation_role:'validation'}}];
  s.rpoCandidates.value = [{id:'native', embodiment_env_id:'rpo_bfm', execution:{provider_id:'ufo'}}];
  s.suiteId.value = 'ufo_validation'; s.source.value = 'native'; s.profile.value = 'validation';
  assert.equal(s.visibleSuites.value.length, 1);
  assert.equal(s.compatibleRpoCandidates.value.length, 1);
  assert.equal(s.body().record_video, true);
  assert.equal(s.body().profile, 'validation');
});

test('UFO reports show physical failure details and velocity metrics', () => {
  const s = setup('EvalReport', ['row', 'isRpo', 'isRpoFlat', 'caseResult'], async () => ({}));
  s.row.value = {gate_results:{receipt:{schema_version:'eb/rpo-bfm-eval/v1'}}};
  assert.equal(s.isRpo.value, true);
  assert.equal(s.isRpoFlat.value, true);
  assert.equal(s.caseResult({status:'task_failure', hard_limit:true}), '硬越限');
});

test('UFO detail reads progress unit from frozen binding and resumes absolute target', () => {
  const s = setup('TrainJobDetail', ['job', 'isUfo', 'resumeCandidate', 'resumeTarget', 'resumeBody'], async () => ({}));
  s.job.value = {params:{execution:{provider_id:'ufo', progress_unit:'learner_joint_update'}}};
  s.resumeCandidate.value = {termination:'target_learner_updates'};
  s.resumeTarget.value = 24;
  assert.equal(s.isUfo.value, true);
  assert.equal(s.resumeBody().target_learner_updates, 24);
});

test('UFO summary reads real RLinf train tags, including zero-update warmup', () => {
  const s = setup('TrainJobDetail', ['metrics', 'ufoSummary'], async () => ({}));
  s.metrics.value = {series: {'train/ufo/learner_joint_updates': [{step: 0, value: 0}]}};
  assert.equal(s.ufoSummary.value.find(x => x.name === 'train/ufo/learner_joint_updates').value, 0);
  s.metrics.value.series['train/ufo/learner_joint_updates'].push({step: 24, value: 24});
  assert.equal(s.ufoSummary.value.find(x => x.name === 'train/ufo/learner_joint_updates').value, 24);
});
