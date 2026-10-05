import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { ref, computed } from 'vue';

const tick = () => new Promise(resolve => setImmediate(resolve));

test('Run route changes discard stale responses; terminal polling refreshes artifacts and cleans up', async () => {
  const source = readFileSync(new URL('../src/views/RunDetail.vue', import.meta.url), 'utf8');
  const script = source.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const requests = new Map(), timers = new Map(), sockets = [], watchers = [];
  let nextTimer = 0;
  const context = vm.createContext({ ref, computed, AbortController, console,
    useRoute: () => ({ params: { id: 'first' } }),
    watch(...args) { watchers.push(args); }, onUnmounted() {},
    api(path) {
      if (path.endsWith('/equivalent-command')) return Promise.resolve({ command: 'actual command' });
      return new Promise(resolve => requests.set(path, resolve));
    },
    setTimeout(fn) { const id = ++nextTimer; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); }, location: { protocol: 'http:', host: 'localhost' },
    WebSocket: class { constructor() { sockets.push(this); } close() { this.closed = true; } },
  });
  const code = ts.transpile(script.replace(/^import .*;$/gm, ''), { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.subject = { data, metrics, logs };`, context);
  const s = context.subject;
  let firstCleanup, secondCleanup;
  watchers[0][1]('first', undefined, fn => { firstCleanup = fn; });
  firstCleanup();
  watchers[0][1]('second', 'first', fn => { secondCleanup = fn; });
  requests.get('/runs/second')({ id: 'second', status: 'running', job_id: 'job_second', gpu_ids: [2] });
  requests.get('/runs/second/metrics')({ series: { actual: [] } });
  requests.get('/runs/second/logs?tail=200')({ lines: ['second run'] });
  await tick();
  requests.get('/runs/first')({ id: 'first', status: 'running' });
  requests.get('/runs/first/metrics')({ series: { stale: [] } });
  requests.get('/runs/first/logs?tail=200')({ lines: ['stale'] });
  await tick();
  assert.equal(s.data.value.id, 'second');
  assert.deepEqual(Array.from(s.logs.value), ['second run']);
  assert.equal(sockets.length, 1);
  assert.equal(timers.size, 1);
  const [timerId, refresh] = timers.entries().next().value;
  timers.delete(timerId);
  const polling = refresh();
  assert.equal(timers.size, 0, 'no overlapping poll is scheduled while a request is pending');
  requests.get('/runs/second')({ id: 'second', status: 'finished', gpu_ids: [2],
    metrics_summary: { eval_report_id: 'ev_actual', checkpoint: { path: '/actual/model' } } });
  await polling;
  assert.equal(s.data.value.metrics_summary.eval_report_id, 'ev_actual');
  assert.equal(s.data.value.metrics_summary.checkpoint.path, '/actual/model');
  assert.equal(timers.size, 0, 'terminal runs stop polling');
  secondCleanup();
  assert.equal(sockets[0].closed, true);
  sockets[0].onmessage({ data: JSON.stringify({ type: 'log', payload: { lines: ['late'] } }) });
  assert.deepEqual(Array.from(s.logs.value), ['second run']);
});
