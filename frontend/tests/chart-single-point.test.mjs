import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

test('a one-sample metric is visible in the shared chart', async () => {
  const source = readFileSync(new URL('../src/components/Chart.vue', import.meta.url), 'utf8');
  const script = source.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  let option;
  const context = vm.createContext({
    defineProps: () => ({ points: [{ step: 1, value: 3.5512028 }] }),
    ref: () => ({ value: {} }),
    onMounted: () => {},
    watch: () => {},
    getEcharts: async () => ({ init: () => ({ setOption: value => { option = value; } }) }),
  });
  const code = ts.transpile(script.replace(/^import .*;$/gm, '').replace('await import("echarts")', 'await getEcharts()'), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  });
  vm.runInContext(`${code}\nglobalThis.drawChart = draw;`, context);
  await context.drawChart();
  assert.equal(option.series[0].showSymbol, true);
  assert.deepEqual(Array.from(option.series[0].data[0]), [1, 3.5512028]);
});
