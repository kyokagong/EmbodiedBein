import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed, createSSRApp } from 'vue';
import { renderToString } from '@vue/server-renderer';

function setup(respond) {
  const calls = [];
  const source = readFileSync(new URL('../src/components/PbfmPanel.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, defineProps: () => ({ model: { id: 'mv_pbfm' } }),
    watch() {}, onMounted() {},
    crypto: { randomUUID: () => 'fixed-key' },
    api: async (path, options) => { calls.push([path, options]); return respond(path, options); },
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + '\nglobalThis.subject = { check, submit, preview, error, lastJob, refresh, motionId };', context);
  return { ...context.subject, calls };
}

test('P-BFM export sends precisely the reviewed hash and stable idempotency key', async () => {
  const s = setup(async path => path.endsWith('/preview') ? { hash: 'sha256:plan', plan: { motion_id: 2 } }
    : path.endsWith('/export') ? { id: 'job_export', status: 'queued' } : { eval_reports: [] });
  s.motionId.value = 2;
  await s.check('export');
  assert.deepEqual(JSON.parse(s.calls[0][1].body), { motion_id: 2 });
  s.motionId.value = 3; // A submit must retain the reviewed motion, never a newer input.
  await s.submit();
  const call = s.calls.find(([path]) => path.endsWith('/export'));
  assert.equal(call[0], '/pbfm/models/mv_pbfm/export');
  assert.deepEqual(JSON.parse(call[1].body), { motion_id: 2, preview_hash: 'sha256:plan' });
  assert.equal(call[1].headers['Idempotency-Key'], 'fixed-key');
  assert.equal(s.lastJob.value.id, 'job_export');
  assert.equal(s.preview.value, null);
});

test('P-BFM evaluation rejects stale/failed previews and surfaces the API error', async () => {
  const s = setup(async () => { throw new Error('depth receipt incomplete'); });
  await s.check('evals');
  await s.submit();
  assert.match(s.error.value, /depth receipt incomplete/);
  assert.equal(s.calls.length, 1);
  assert.equal(s.preview.value, null);
});

test('P-BFM report renders indexed videos, depth, traces and plots for successful and failed cases', async () => {
  const template = readFileSync(new URL('../src/components/PbfmPanel.vue', import.meta.url), 'utf8')
    .match(/<template>([\s\S]*?)<\/template>/)[1];
  const status = { export_url: '/policy.zip', export_manifest_url: '/manifest.json', eval_reports: [{
    id: 'report_pbfm', qualification: 'diagnostic', cases_url: '/cases.json',
    media_index_url: '/media-index.json', plot_url: '/summary.svg',
    cases: [true, false].map((success, index) => ({ case_id: `case_${index}`, success,
      depth_condition: index ? 'occluded' : 'normal', video_url: `/case_${index}.mp4`,
      depth_url: `/depth_${index}.mp4`, depth_data_url: `/depth_${index}.npz`, trace_url: `/trace_${index}.npz` })),
  }] };
  const app = createSSRApp({ template, setup: () => ({ status, busy: false, error: '', preview: null,
    lastJob: null, action: 'export', motionId: 0, check() {}, refresh() {}, submit() {}, mediaErrors: {} }) });
  app.component('router-link', { props: ['to'], template: '<a :href="to"><slot /></a>' });
  const html = await renderToString(app);
  for (const path of ['/policy.zip', '/manifest.json', '/cases.json', '/media-index.json', '/summary.svg',
    '/case_0.mp4', '/case_1.mp4', '/depth_0.mp4', '/depth_1.mp4', '/depth_0.npz', '/depth_1.npz', '/trace_0.npz', '/trace_1.npz']) {
    assert.ok(html.includes(path), `missing evidence link: ${path}`);
  }
  assert.equal((html.match(/<video /g) || []).length, 4);
  assert.match(html, /未通过/);
});

test('incomplete evaluation is not presented as measured quality failure', async () => {
  const template = readFileSync(new URL('../src/components/PbfmPanel.vue', import.meta.url), 'utf8')
    .match(/<template>([\s\S]*?)<\/template>/)[1];
  for (const complete of [false, true]) {
    const status = { eval_reports: [{ id: 'ev_depth', gate_results: { validity: complete ? 'complete' : 'incomplete' },
      metrics: { valid: complete ? 32 : 0, planned: 32, cases: 16, normal_success_rate: 0,
        occluded_success_rate: 0, quality_passed: false }, cases: [] }] };
    const app = createSSRApp({ template, setup: () => ({ status, busy: false, error: '', preview: null,
      lastJob: null, action: 'export', motionId: 0, check() {}, refresh() {}, submit() {}, mediaErrors: {} }) });
    app.component('router-link', { props: ['to'], template: '<a :href="to"><slot /></a>' });
    const html = await renderToString(app);
    if (complete) assert.match(html, /模型质量：未通过/);
    else { assert.match(html, /评估未完成/); assert.doesNotMatch(html, /模型质量：未通过/); }
  }
});
