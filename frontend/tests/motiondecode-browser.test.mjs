import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { computed, reactive, ref, watch } from 'vue';

const ep = { episode_id: 'walk', path: 'motions/walking.csv', length: 8000 };
const preview = { columns: ['root_x', 'root_y'], frames: 8000, start_frame: 0, end_frame: 7200,
  source_fps: 120, fps: 30, video_frames: 1800, video_url: '/api/v1/video',
  table_base_url: '/api/v1/table', table_pages: 80, table_page_size: 100 };
const tick = () => new Promise(resolve => setImmediate(resolve));
function setup(handler, derived = true) {
  const calls = [], navigations = [], emitted = [], timers = new Map(), stops = [];
  let cleanup, timerId = 0;
  const props = reactive({ dataset: { id: 'ds_motion', source_uri: 'CMRobot/MotionDecode',
    meta: derived ? { operation: 'motiondecode_valid_view_v1' } : {} } });
  const source = readFileSync(new URL('../src/components/MotionDecodeBrowser.vue', import.meta.url), 'utf8')
    .match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
  const context = vm.createContext({ ref, computed, defineProps: () => props,
    defineEmits: () => (...args) => emitted.push(args),
    useRouter: () => ({ push(path) { navigations.push(path); } }),
    watch(...args) { const stop = watch(...args); stops.push(stop); return stop; },
    onUnmounted(fn) { cleanup = fn; },
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    api: async (path, init) => { calls.push([path, init]); return handler(path, init); },
  });
  vm.runInContext(ts.transpile(source.replace(/^import .*;$/gm, ''), {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  }) + '\nglobalThis.subject = { query, episodes, cursor, loading, listError, selected, startFrame, job, preview, previewError, previewLoading, table, tablePage, tableError, error, creating, openValidView, retryValidation, loadEpisodes, loadTable, generate };', context);
  return { ...context.subject, props, calls, timers, navigations, emitted,
    async poll() {
      const [id, timer] = timers.entries().next().value;
      timers.delete(id);
      await timer.fn();
    },
    stop() { cleanup(); stops.forEach(stop => stop()); },
  };
}
function reply(path) {
  if (path.includes('/episodes?')) return { items: [ep], total: 1, next_cursor: null };
  if (path.startsWith('/table/')) return { columns: preview.columns, rows: [[1, 2]], frame_start: 0, total_frames: 8000 };
  return { id: 'job_preview', status: 'succeeded', preview };
}

test('raw source opens an idempotent derived view without indexing raw episodes', async () => {
  const s = setup(async () => ({ id: 'ds_validated' }), false);
  assert.equal(s.calls.length, 0);
  await s.openValidView();
  assert.equal(s.calls[0][0], '/datasets/ds_motion/motiondecode/valid-view');
  assert.equal(s.calls[0][1].body, '{}');
  assert.deepEqual(s.navigations, ['/data/ds_validated']);
  s.props.dataset.meta.valid_view_id = 'ds_existing';
  await s.openValidView();
  assert.equal(s.calls.length, 1);
  assert.equal(s.navigations[1], '/data/ds_existing');
  s.stop();
});

test('search and paging use server query; late list responses cannot replace newer results', async () => {
  let old;
  const s = setup(path => path.includes('q=walking') ? new Promise(done => { old = done; }) : reply(path));
  await tick();
  s.query.value = 'walking';
  const pending = s.loadEpisodes();
  s.query.value = 'turn & jump';
  await s.loadEpisodes(50);
  assert.match(s.calls.at(-1)[0], /cursor=50&limit=50&q=turn%20%26%20jump$/);
  old({ items: [], total: 0, next_cursor: null });
  await pending;
  assert.equal(s.episodes.value.total, 1);
  assert.equal(s.cursor.value, 50);
  s.stop();
});

test('preview polls until succeeded and fetches only requested source table pages', async () => {
  const s = setup(path => path.endsWith('/previews') ? { id: 'job_preview', status: 'queued' } : reply(path));
  await tick();
  await s.generate(ep, 0);
  assert.equal(s.timers.size, 1);
  assert.deepEqual(JSON.parse(s.calls.at(-1)[1].body), { path: ep.path, start_frame: 0 });
  await s.poll();
  assert.equal(s.previewLoading.value, false);
  assert.equal(s.preview.value.video_url, preview.video_url);
  assert.equal(s.calls.at(-1)[0], '/table/000000.json');
  await s.loadTable(2);
  assert.equal(s.calls.at(-1)[0], '/table/000002.json');
  assert.equal(s.tablePage.value, 2);
  assert.equal(s.timers.size, 0);
  await s.generate(ep, 7200);
  assert.equal(JSON.parse(s.calls.at(-1)[1].body).start_frame, 7200);
  s.stop();
});

test('failed preview and source table requests show retryable errors', async () => {
  let fail = true, tableFail = true;
  const s = setup(path => {
    if (path.endsWith('/previews') && fail) return { id: 'job_failed', status: 'failed', error: 'CSV is invalid' };
    if (path.startsWith('/table/') && tableFail) throw new Error('table unavailable');
    return reply(path);
  });
  await tick();
  await s.generate(ep, 0);
  assert.equal(s.previewError.value, 'CSV is invalid');
  assert.equal(s.previewLoading.value, false);
  fail = false;
  await s.generate(ep, 0);
  assert.equal(s.previewError.value, '');
  assert.equal(s.tableError.value, 'table unavailable');
  tableFail = false;
  await s.loadTable(2);
  assert.equal(s.tableError.value, '');
  assert.equal(s.table.value.rows[0][1], 2);
  assert.equal(s.tablePage.value, 2);
  s.stop();
});

test('switching motions rejects stale preview responses; unmount cancels polling and pending tables', async () => {
  let resolveOld, resolveTable;
  const s = setup((path, init) => {
    if (path.endsWith('/previews')) {
      if (JSON.parse(init.body).path === ep.path) return new Promise(done => { resolveOld = done; });
      return { id: 'new_job', status: 'running' };
    }
    return reply(path);
  });
  await tick();
  const old = s.generate(ep, 0);
  await s.generate({ ...ep, path: 'motions/new.csv' }, 0);
  resolveOld({ id: 'old_job', status: 'succeeded', preview });
  await old;
  assert.equal(s.job.value.id, 'new_job');
  assert.equal(s.preview.value, null);
  assert.equal(s.timers.size, 1);
  s.stop();
  assert.equal(s.timers.size, 0);

  const t = setup(path => path.startsWith('/table/') ? new Promise(done => { resolveTable = done; }) : reply(path));
  await tick();
  const pending = t.generate(ep, 0);
  await tick();
  t.stop();
  resolveTable({ columns: ['x'], rows: [[999]], frame_start: 0 });
  await pending;
  assert.equal(t.table.value, null);
});

test('route dataset changes discard in-flight responses and invalid source frames never submit', async () => {
  let resolveOld;
  const s = setup((path) => path.includes('/ds_motion/episodes?')
    ? new Promise(done => { resolveOld = done; }) : reply(path));
  s.props.dataset = { ...s.props.dataset, id: 'ds_other' };
  await tick();
  resolveOld({ items: [], total: 999, next_cursor: null });
  await tick();
  assert.equal(s.episodes.value.total, 1);
  const count = s.calls.length;
  await s.generate(ep, -1);
  assert.equal(s.calls.length, count);
  assert.match(s.previewError.value, /整数/);
  s.stop();
});

test('derived validation retries through the parent, refreshes on success and preserves errors', async () => {
  let fail = false;
  const s = setup(path => {
    if (path.endsWith('/valid-view') && fail) throw new Error('validation unavailable');
    return reply(path);
  });
  await tick();
  s.props.dataset.status = 'failed';
  s.props.dataset.meta.parent_dataset_id = 'ds_raw';
  await s.retryValidation();
  assert.equal(s.calls.at(-1)[0], '/datasets/ds_raw/motiondecode/valid-view');
  assert.equal(s.calls.at(-1)[1].body, '{}');
  assert.deepEqual(s.emitted, [['refresh']]);
  assert.equal(s.creating.value, false);
  fail = true;
  await s.retryValidation();
  assert.equal(s.error.value, 'validation unavailable');
  assert.equal(s.emitted.length, 1);
  delete s.props.dataset.meta.parent_dataset_id;
  const count = s.calls.length;
  await s.retryValidation();
  assert.equal(s.calls.length, count);
  s.stop();
});

test('validation completion reloads episodes while preserving the same dataset preview', async () => {
  let queued = true;
  const s = setup(path => queued && path.includes('/episodes?')
    ? { items: [], total: 0, next_cursor: null } : reply(path));
  await tick();
  assert.equal(s.episodes.value.total, 0);
  s.props.dataset.status = 'queued';
  s.selected.value = ep;
  s.preview.value = preview;
  const shownPreview = s.preview.value;
  await tick();
  const calls = s.calls.length;
  queued = false;
  s.props.dataset.status = 'validated';
  await tick();
  assert.equal(s.calls.length, calls + 1);
  assert.equal(s.episodes.value.total, 1);
  assert.equal(s.preview.value, shownPreview);
  assert.equal(s.selected.value.path, ep.path);
  s.stop();
});

test('infrastructure failure, timeout and blocked preview jobs stop polling immediately', async () => {
  for (const status of ['infra_failed', 'timeout', 'blocked']) {
    const s = setup(path => path.endsWith('/previews')
      ? { id: 'job_terminal', status, error: `terminal: ${status}` } : reply(path));
    await tick();
    await s.generate(ep, 0);
    assert.equal(s.previewLoading.value, false);
    assert.equal(s.previewError.value, `terminal: ${status}`);
    assert.equal(s.timers.size, 0);
    s.stop();
  }
});
