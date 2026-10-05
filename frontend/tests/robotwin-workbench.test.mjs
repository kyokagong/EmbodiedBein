import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('dataset page uses the bounded RoboTwin catalog and acquisition endpoints', () => {
  const source = readFileSync(new URL('../src/views/DatasetNew.vue', import.meta.url), 'utf8');
  assert.match(source, /\/dataset-sources\/robotwin/);
  assert.match(source, /\/datasets\/robotwin/);
  assert.match(source, /profile_id/);
  assert.match(source, /native_embodiment|embodiment/);
});

test('dataset detail displays native embodiment and compatibility evidence', () => {
  const source = readFileSync(new URL('../src/views/DatasetDetail.vue', import.meta.url), 'utf8');
  assert.match(source, /native_embodiment/);
  assert.match(source, /compatibility/);
  assert.match(source, /training_ready/);
});

test('model page exposes the six-model compatibility matrix', () => {
  const source = readFileSync(new URL('../src/views/ModelList.vue', import.meta.url), 'utf8');
  assert.match(source, /dataset-sources\/robotwin/);
  assert.match(source, /cross_embodiment/);
  assert.match(source, /direct_embodiments/);
  assert.match(source, /blocked_reason/);
});

test('RoboTwin evaluation exposes bounded multi-GPU controls', () => {
  const source = readFileSync(new URL('../src/views/EvalNew.vue', import.meta.url), 'utf8');
  assert.match(source, /resourceVariant\.parallel_envs|resource_variant_id/);
  assert.match(source, /模型副本/);
  assert.match(source, /总并发环境/);
  assert.match(source, /推理 micro-batch/);
  assert.match(source, /Headless/);
  assert.match(source, /离屏三相机/);
  assert.match(source, /视频关闭/);
  assert.match(source, /未通过 probe/);
});

test('RoboTwin evaluation uses server profiles and bounded container memory variants', () => {
  const source = readFileSync(new URL('../src/views/EvalNew.vue', import.meta.url), 'utf8');
  assert.match(source, /runtime-profiles/);
  assert.match(source, /runtime_profile_id/);
  assert.match(source, /resource_variant_id/);
  assert.match(source, /高级配置/);
  assert.match(source, /容器内存/);
  assert.match(source, /共享内存/);
  assert.doesNotMatch(source, /memory_gb\s*:/);
});
