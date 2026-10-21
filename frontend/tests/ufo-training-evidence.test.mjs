import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { computed, ref } from 'vue';

function setup(row, evidenceResponse) {
  const calls = [];
  const source = readFileSync(new URL('../src/views/TrainJobDetail.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, AbortController,
    useRoute: () => ({ params: { id: 'trn_native' } }), useRouter: () => ({ push() {} }),
    onUnmounted() {}, watch() {}, clearTimeout() {}, window: { setTimeout() {} },
    api: async path => {
      calls.push(path);
      if (path.endsWith('/evidence')) return evidenceResponse();
      if (path.endsWith('/checkpoints')) return { items: [] };
      if (path.endsWith('/resume-candidate')) return { available: false };
      if (path.includes('/metrics')) return { series: { fb_loss: [{ step: 112, value: 1 }] } };
      if (path.includes('/logs')) return { lines: [] };
      return row;
    },
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + '\nglobalThis.subject = { load, trainingEvidence, metrics, err, experimentTarget, job };', context);
  return { ...context.subject, calls };
}

test('completed native training loads historical evidence without hiding metric curves', async () => {
  const evidence = { available: true, run_id: 'run_native', files: [{ path: 'analysis/training-1.png', url: '/verified' }] };
  const s = setup({ kind: 'native_rl', status: 'succeeded', run_id: 'run_native' }, async () => evidence);
  await s.load();
  assert.equal(s.trainingEvidence.value.run_id, 'run_native');
  assert.equal(s.trainingEvidence.value.files[0].url, '/verified');
  assert.equal(s.metrics.value.series.fb_loss[0].step, 112);
  assert.equal(s.calls.filter(path => path.endsWith('/evidence')).length, 1);
});

test('already indexed summaries avoid redundant reads; active and other training keep existing polling', async () => {
  for (const row of [
    { kind: 'native_rl', status: 'succeeded', metrics_summary: { training_evidence: { available: true, run_id: 'run_old' } } },
    { kind: 'native_rl', status: 'running' },
    { kind: 'ppo', status: 'succeeded' },
  ]) {
    const s = setup(row, () => { throw new Error('unexpected evidence read'); });
    await s.load();
    assert.equal(s.calls.some(path => path.endsWith('/evidence')), false);
    assert.equal(s.err.value, '');
  }
});

test('incomplete evidence is visible while checkpoints and live curves still load', async () => {
  const s = setup({ kind: 'native_rl', status: 'succeeded', run_id: 'run_native' }, () => {
    throw new Error('原生训练证据不完整：hash differs');
  });
  await s.load();
  assert.match(s.err.value, /训练证据.*hash differs/);
  assert.equal(s.trainingEvidence.value, null);
  assert.equal(s.metrics.value.series.fb_loss.length, 1);
});

test('finished 20000 interval keeps its count and exposes the continuing 100000 goal', async () => {
  const row = { kind: 'native_rl', status: 'succeeded',
    params: { native_train_plan: { target_updates: 20000 } },
    continuations: [{ id: 'trn_next', status: 'running',
      params: { native_train_plan: { target_updates: 30000, experiment_target_updates: 100000 } } }],
  };
  const s = setup(row, async () => ({ available: true }));
  await s.load();
  assert.equal(s.experimentTarget.value, 100000);
  assert.equal(s.job.value.params.native_train_plan.target_updates, 20000);
  assert.equal(s.job.value.continuations[0].id, 'trn_next');
  s.job.value = { params: { native_train_plan: { target_updates: 30000, experiment_target_updates: 100000 } } };
  assert.equal(s.experimentTarget.value, 100000);
  s.job.value = { params: { native_train_plan: { target_updates: 20000 } } };
  assert.equal(s.experimentTarget.value, undefined);
});
