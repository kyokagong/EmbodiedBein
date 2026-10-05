import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { computed, nextTick, ref, watch } from "vue";

test("shared list behavior filters, sorts and clamps pagination", async () => {
  const source = readFileSync(new URL("../src/composables/useListView.ts", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "")
    .replace(/^export /gm, "");
  const context = vm.createContext({ computed, ref, watch });
  const code = ts.transpile(source, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInContext(`${code}\nglobalThis.useListView = useListView;`, context);

  const rows = ref(Array.from({ length: 23 }, (_, i) => ({ name: `任务 ${i + 1}`, size: 23 - i })));
  const list = context.useListView(rows, [
    { key: "name", label: "名称", value: (row) => row.name },
    { key: "size", label: "规模", value: (row) => row.size },
  ], (row) => row.name, [
    { key: "name", label: "名称", type: "text", value: (row) => row.name },
    { key: "band", label: "状态", type: "multi", value: (row) => row.size > 20 ? "large" : "small" },
  ]);
  list.pageSize.value = 10;
  await nextTick();
  assert.equal(list.pageCount.value, 3);
  list.page.value = 3;
  assert.equal(list.visibleItems.value.length, 3);

  list.sortKey.value = "size";
  list.sortDirection.value = "desc";
  await nextTick();
  assert.equal(list.page.value, 1);
  assert.equal(list.visibleItems.value[0].size, 23);

  list.query.value = "任务 2";
  await nextTick();
  assert.equal(list.total.value, 5);
  assert.equal(list.pageCount.value, 1);

  list.query.value = "";
  list.textFilters.value = { name: "任务 1" };
  list.multiFilters.value = { band: ["small"] };
  await nextTick();
  assert.equal(list.total.value, 10);
  assert.deepEqual(Array.from(list.filterOptions.value.band), ["large", "small"]);
  assert.equal(list.activeFilterCount.value, 2);
  list.clearFilters();
  await nextTick();
  assert.equal(list.total.value, 23);
  assert.equal(list.activeFilterCount.value, 0);
});

test("all requested list pages use the shared controls", () => {
  for (const file of ["TrainList.vue", "ModelList.vue", "EvalJobs.vue", "DatasetList.vue", "EmbodimentList.vue"]) {
    const source = readFileSync(new URL(`../src/views/${file}`, import.meta.url), "utf8");
    assert.match(source, /<ListControls/);
    assert.match(source, /list\.visibleItems\.value/);
  }
});
