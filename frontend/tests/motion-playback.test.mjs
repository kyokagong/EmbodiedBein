import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('../src/utils/motionPlayback.ts', import.meta.url), 'utf8');
const context = vm.createContext({});
vm.runInContext(ts.transpile(source.replace(/export /g, ''), { target: ts.ScriptTarget.ES2022 }) +
  '\nglobalThis.helpers = { frameAtTime, motionPose };', context);

test('playback uses the saved timestamps and stops at the clip boundary', () => {
  const { frameAtTime } = context.helpers;
  assert.equal(frameAtTime([2, 2.02, 2.04], 1), 0);
  assert.equal(frameAtTime([2, 2.02, 2.04], 2.019), 0);
  assert.equal(frameAtTime([2, 2.02, 2.04], 2.02), 1);
  assert.equal(frameAtTime([2, 2.02, 2.04], 9), 2);
});

test('pose maps by joint name, preserves raw angles and height, and centers only horizontal travel', () => {
  const pose = context.helpers.motionPose({ joint_order: ['right', 'left'], q: [[0, 0], [9, -8]],
    root_pos: [[10, 20, .7], [11, 18, .8]], root_quat: [[1, 0, 0, 0], [0, 0, 0, 1]] }, 1);
  assert.equal(pose.joints.right, 9);
  assert.equal(pose.joints.left, -8);
  assert.deepEqual(Array.from(pose.position), [1, -2, .8]);
  assert.deepEqual(Array.from(pose.quaternion), [0, 0, 0, 1]);
});

test('the shared chart reuses and disposes its instance during playback', async () => {
  const vue = readFileSync(new URL('../src/components/Chart.vue', import.meta.url), 'utf8');
  const script = vue.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  let init = 0, dispose = 0, cleanup;
  const chartContext = vm.createContext({ defineProps: () => ({ points: [], playhead: 0 }),
    ref: () => ({ value: {} }), onMounted: () => {}, watch: () => {},
    onUnmounted: fn => { cleanup = fn; },
    getEcharts: async () => ({ init: () => { init++; return { setOption() {}, dispose() { dispose++; } }; } }) });
  vm.runInContext(ts.transpile(script.replace(/^import .*;$/gm, '').replace('await import("echarts")', 'await getEcharts()'),
    { target: ts.ScriptTarget.ES2022 }) + '\nglobalThis.redraw = draw;', chartContext);
  await chartContext.redraw();
  await chartContext.redraw();
  assert.equal(init, 1);
  cleanup();
  assert.equal(dispose, 1);
});
