import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

function component(file, names, respond) {
  const watchers = [], route = { params: { id: 'mv_export' } };
  const source = readFileSync(new URL(`../src/views/${file}.vue`, import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed,
    onMounted() {}, watch(...args) { watchers.push(args); },
    useRoute: () => route, useRouter: () => ({}), useListView: () => ({}), api: respond,
  });
  const code = ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  });
  vm.runInContext(`${code}\nglobalThis.subject = { ${names.join(', ')} };`, context);
  return { ...context.subject, watchers, route };
}

const trained = { id: 'mv_trained', name: 'native training', train_page_id: 'trn_source',
  train_job_name: 'native RPO 112 updates', tags: ['rpo-v2', 'ufo-native'],
  card: { schema_version: 'eb/rpo-ufo-native-trained-model/v1', training_provenance: 'platform',
    train_job_id: 'job_source', counts: { optimizer_steps: 112 }, package_path: '/frozen/model' } };
const prompted = { id: 'mv_prompted', tags: ['rpo-v2', 'ufo-native'],
  card: { schema_version: 'eb/rpo-ufo-native-model/v1', training_provenance: 'platform',
    source_model_id: trained.id, readiness: 'prompt_derived', package_path: '/frozen/prompted' } };
const exported = { id: 'mv_export', tags: ['rpo-v2', 'ufo-native'],
  card: { schema_version: 'eb/rpo-ufo-native-export/v1', training_provenance: 'platform',
    source_model_id: prompted.id, package_path: '/frozen/exported' } };

test('native trained and derived model rows retain training pages and actual update counts', async () => {
  const s = component('ModelList', ['load', 'items'], async path => path === '/models'
    ? { versions: [trained, prompted, exported], items: [] } : { items: [], models: [] });
  await s.load();
  const rows = Array.from(s.items.value);
  for (const row of rows) {
    assert.equal(row.native, true);
    assert.equal(row.train_page_id, 'trn_source');
    assert.equal(row.train_job_name, 'native RPO 112 updates');
    assert.equal(row.step, 112);
    assert.doesNotMatch(row.kind, /WBC|PPO/);
  }
  assert.match(rows[0].kind, /FB-CPR-Aux/);
  assert.equal(rows[2].path, '/frozen/exported');
});

test('native model detail follows prompt/export lineage to its original training page', async () => {
  const models = new Map([trained, prompted, exported].map(model => [model.id, model]));
  const s = component('ModelDetail', ['data', 'isNative', 'trainingSource', 'trainingPage', 'trainingName'], async path => {
    if (path.endsWith('/lineage')) return { edges: [], nodes: [] };
    return models.get(path.split('/').at(-1));
  });
  await s.watchers.find(([source]) => typeof source === 'function')[1]('mv_export');
  assert.equal(s.isNative.value, true);
  assert.equal(s.data.value.id, 'mv_export');
  assert.equal(s.trainingSource.value.id, 'mv_trained');
  assert.equal(s.trainingPage.value, 'trn_source');
  assert.equal(s.trainingName.value, 'native RPO 112 updates');
});

test('external native imports keep no invented platform training lineage', async () => {
  const external = { id: 'mv_external', card: { schema_version: 'eb/rpo-ufo-native-model/v1',
    training_provenance: 'external_declared', package_path: '/imported' } };
  const s = component('ModelList', ['load', 'items'], async path => path === '/models'
    ? { versions: [external], items: [] } : { items: [], models: [] });
  await s.load();
  assert.equal(s.items.value[0].train_page_id, undefined);
  assert.equal(s.items.value[0].step, undefined);
  assert.equal(s.items.value[0].kind, 'UFO/RPO · 原生模型');
});

test('P-BFM trained and exported policies keep their own family and update count', async () => {
  const pbfm = { ...trained, id: 'mv_pbfm', card: { ...trained.card, schema_version: 'eb/rpo-pbfm-trained-model/v1' } };
  const policy = { id: 'mv_pbfm_policy', tags: ['rpo-v2'],
    card: { schema_version: 'eb/rpo-pbfm-policy/v1', source_model_id: pbfm.id, package_path: '/frozen/pbfm-policy' } };
  const s = component('ModelList', ['load', 'items'], async path => path === '/models'
    ? { versions: [pbfm, policy], items: [] } : { items: [], models: [] });
  await s.load();
  for (const row of s.items.value) {
    assert.equal(row.native, true);
    assert.match(row.kind, /P-BFM/);
    assert.doesNotMatch(row.kind, /UFO|WBC|PPO/);
    assert.equal(row.step, 112);
    assert.equal(row.train_page_id, 'trn_source');
  }
});
