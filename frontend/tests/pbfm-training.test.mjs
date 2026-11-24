import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

function setup(file, names, api = async () => ({})) {
  const source = readFileSync(new URL(`../src/views/${file}.vue`, import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, api, AbortController,
    onMounted() {}, onUnmounted() {}, watch() {}, clearTimeout() {},
    window: { setTimeout() {} }, useRoute: () => ({ params: { id: 'mv_pbfm' } }),
    useRouter: () => ({ push() {} }),
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + `\nglobalThis.subject = { ${names.join(',')} };`, context);
  return context.subject;
}

test('P-BFM consumes its validated training split and sends only P-BFM options', () => {
  const s = setup('TrainNew', ['form', 'templates', 'embodiments', 'datasets', 'compatibleDatasets', 'payload']);
  const env = 'ee_rpo_v2_pbfm_depth_23dof_v1';
  s.templates.value = [{ id: 'rpo_pbfm_fb_depth_native', kind: 'native_rl', embodiment_env_id: env }];
  s.embodiments.value = [{ id: 'emb_roboparty_rpo', envs: [{ id: env }] }];
  Object.assign(s.form.value, { template_id: 'rpo_pbfm_fb_depth_native', embodiment_id: 'emb_roboparty_rpo' });
  const good = { id: 'ds_pbfm', embodiment_env_id: env, usage: 'train', status: 'validated',
    source_kind: 'pbfm_rpo_motion', validation: { scope: 'pbfm_rpo_motion_v1', training_ready: true } };
  s.datasets.value = [good, { ...good, id: 'holdout', usage: 'eval' },
    { ...good, id: 'ufo', source_kind: 'ufo_native_motion' },
    { ...good, id: 'unverified', validation: { scope: 'pbfm_rpo_motion_v1', training_ready: false } }];
  assert.deepEqual(Array.from(s.compatibleDatasets.value, x => x.id), ['ds_pbfm']);
  const payload = s.payload();
  assert.equal(payload.pbfm_training.target_updates, 112);
  assert.equal(payload.pbfm_training.buffer_size, 262144);
  assert.equal('native_training' in payload, false);
});

test('blocked P-BFM dependency state cannot enable submission', () => {
  const s = setup('TrainNew', ['form', 'templates', 'environments', 'missingInput']);
  Object.assign(s.form.value, { name: 'terrain', model_name: 'depth', embodiment_id: 'emb_roboparty_rpo',
    template_id: 'rpo_pbfm_fb_depth_native', environment_id: 'profile', dataset_ids: ['train'] });
  s.templates.value = [{ id: 'rpo_pbfm_fb_depth_native', kind: 'native_rl', ready: false }];
  s.environments.value = [{ id: 'profile', template_id: 'rpo_pbfm_fb_depth_native', usage: 'train', status: 'active' }];
  assert.equal(s.missingInput.value, true);
});

test('P-BFM model is identified without activating the UFO panel', () => {
  const s = setup('ModelDetail', ['data', 'isNative', 'isPbfm']);
  s.data.value = { id: 'mv_pbfm', card: { schema_version: 'eb/rpo-pbfm-trained-model/v1' } };
  assert.equal(s.isPbfm.value, true);
  assert.equal(s.isNative.value, false);
});
