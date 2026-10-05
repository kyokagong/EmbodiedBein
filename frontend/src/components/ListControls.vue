<script setup lang="ts">
import type { ListFilter, SortOption } from "../composables/useListView";

const props = defineProps<{
  query: string;
  sortKey: string;
  sortDirection: "asc" | "desc";
  page: number;
  pageSize: number;
  total: number;
  pageCount: number;
  sortOptions: SortOption<any>[];
  filters?: ListFilter<any>[];
  textFilters?: Record<string, string>;
  multiFilters?: Record<string, string[]>;
  filterOptions?: Record<string, string[]>;
  activeFilterCount?: number;
}>();

const emit = defineEmits<{
  "update:query": [value: string];
  "update:sortKey": [value: string];
  "update:sortDirection": [value: "asc" | "desc"];
  "update:page": [value: number];
  "update:pageSize": [value: number];
  "update:textFilters": [value: Record<string, string>];
  "update:multiFilters": [value: Record<string, string[]>];
  "clear-filters": [];
}>();

function setTextFilter(key: string, value: string) {
  emit("update:textFilters", { ...props.textFilters, [key]: value });
}

function toggleMultiFilter(key: string, value: string, checked: boolean) {
  const selected = props.multiFilters?.[key] || [];
  emit("update:multiFilters", { ...props.multiFilters,
    [key]: checked ? [...selected, value] : selected.filter((item) => item !== value) });
}
</script>

<template>
  <div class="list-tools" role="search" aria-label="列表过滤和排序">
    <div class="list-controls">
      <input
        class="list-search"
        type="search"
        :value="query"
        placeholder="输入关键字过滤"
        aria-label="输入关键字过滤"
        @input="emit('update:query', ($event.target as HTMLInputElement).value)"
      >
      <label>
        <span>排序</span>
        <select :value="sortKey" aria-label="排序字段" @change="emit('update:sortKey', ($event.target as HTMLSelectElement).value)">
          <option v-for="option in sortOptions" :key="option.key" :value="option.key">{{ option.label }}</option>
        </select>
      </label>
      <button
        type="button"
        :aria-label="sortDirection === 'asc' ? '当前升序，点击切换为降序' : '当前降序，点击切换为升序'"
        @click="emit('update:sortDirection', sortDirection === 'asc' ? 'desc' : 'asc')"
      >{{ sortDirection === "asc" ? "↑ 升序" : "↓ 降序" }}</button>
      <label>
        <span>每页</span>
        <select :value="pageSize" aria-label="每页数量" @change="emit('update:pageSize', Number(($event.target as HTMLSelectElement).value))">
          <option :value="10">10</option><option :value="20">20</option><option :value="50">50</option>
        </select>
      </label>
      <span class="muted xs">共 {{ total }} 条</span>
      <div class="list-pager" aria-label="分页">
        <button type="button" :disabled="page <= 1" @click="emit('update:page', page - 1)">上一页</button>
        <span class="xs">{{ page }} / {{ pageCount }}</span>
        <button type="button" :disabled="page >= pageCount" @click="emit('update:page', page + 1)">下一页</button>
      </div>
    </div>
    <details v-if="filters?.length" class="list-filter-box">
      <summary>字段过滤 <span v-if="activeFilterCount" class="badge">{{ activeFilterCount }}</span></summary>
      <div class="list-filter-grid">
        <template v-for="filter in filters" :key="filter.key">
          <label v-if="filter.type === 'text'" class="list-field-filter">
            <span>{{ filter.label }}</span>
            <input :value="textFilters?.[filter.key] || ''" :placeholder="`过滤${filter.label}`" @input="setTextFilter(filter.key, ($event.target as HTMLInputElement).value)">
          </label>
          <fieldset v-else class="list-multi-filter">
            <legend>{{ filter.label }}</legend>
            <label v-for="option in filterOptions?.[filter.key] || []" :key="option">
              <input type="checkbox" :checked="multiFilters?.[filter.key]?.includes(option)" @change="toggleMultiFilter(filter.key, option, ($event.target as HTMLInputElement).checked)">
              <span>{{ option }}</span>
            </label>
          </fieldset>
        </template>
      </div>
      <button v-if="activeFilterCount || query" type="button" @click="emit('clear-filters')">清除全部过滤</button>
    </details>
  </div>
</template>
