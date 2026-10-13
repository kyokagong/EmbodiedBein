import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref } from 'vue';

function setup(plan, confirm = true, failure = false) {
  const calls = [], events = [], confirmations = [];
  const source = readFileSync(new URL('../src/components/DeleteButton.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref,
    defineProps: () => ({ path: '/train/trn_test', name: 'test' }),
    defineEmits: () => (name) => events.push(name),
    window: { confirm: (message) => { confirmations.push(message); return confirm; } },
    api: async (path, init) => {
      calls.push([path, init?.method]);
      if (!init) return plan;
      if (failure) throw new Error('{"detail":"cleanup failed; retry"}');
      return { ok: true };
    },
  });
  const code = ts.transpile(source.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.subject = { remove, busy, error };`, context);
  return { ...context.subject, calls, events, confirmations };
}

test('blocked or cancelled deletion never sends DELETE', async () => {
  const blocked = setup({ blockers: [{ id: 'mv_used' }] });
  await blocked.remove();
  assert.equal(blocked.calls.length, 1);
  assert.match(blocked.error.value, /mv_used/);
  assert.equal(blocked.confirmations.length, 0);
  const cancelled = setup({ paths: ['/managed/run_test'] }, false);
  await cancelled.remove();
  assert.equal(cancelled.calls.length, 1);
  assert.equal(cancelled.events.length, 0);
});

test('confirmed cleanup shows paths and only emits deleted after success', async () => {
  const success = setup({ paths: ['/managed/run_test'], preserved_paths: ['/external/source'] });
  await success.remove();
  assert.match(success.confirmations[0], /\/managed\/run_test/);
  assert.match(success.confirmations[0], /\/external\/source/);
  assert.deepEqual(success.calls[1], ['/train/trn_test', 'DELETE']);
  assert.deepEqual(success.events, ['deleted']);
  const failed = setup({}, true, true);
  await failed.remove();
  assert.deepEqual(failed.events, []);
  assert.equal(failed.error.value, 'cleanup failed; retry');
  assert.equal(failed.busy.value, false);
});
