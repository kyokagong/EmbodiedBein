import { computed, ref, watch, type Ref } from "vue";

export type SortOption<T> = {
  key: string;
  label: string;
  value: (item: T) => string | number | null | undefined;
};

export type ListFilter<T> = {
  key: string;
  label: string;
  type: "text" | "multi";
  value: (item: T) => string | null | undefined;
};

export function useListView<T>(
  source: Ref<T[]>,
  sortOptions: SortOption<T>[],
  searchable: (item: T) => string,
  filters: ListFilter<T>[] = [],
  defaultSort = sortOptions[0]?.key || "",
  defaultDirection: "asc" | "desc" = "asc",
) {
  const query = ref("");
  const sortKey = ref(defaultSort);
  const sortDirection = ref<"asc" | "desc">(defaultDirection);
  const page = ref(1);
  const pageSize = ref(20);
  const textFilters = ref<Record<string, string>>({});
  const multiFilters = ref<Record<string, string[]>>({});

  const filterOptions = computed(() => Object.fromEntries(filters
    .filter((filter) => filter.type === "multi")
    .map((filter) => [filter.key, [...new Set(source.value.map(filter.value).filter(Boolean).map(String))]
      .sort((a, b) => a.localeCompare(b, "zh-CN", { numeric: true }))])));
  const activeFilterCount = computed(() =>
    Object.values(textFilters.value).filter((value) => value.trim()).length
    + Object.values(multiFilters.value).filter((values) => values.length).length);

  const arranged = computed(() => {
    const needle = query.value.trim().toLocaleLowerCase();
    const searched = needle
      ? source.value.filter((item) => searchable(item).toLocaleLowerCase().includes(needle))
      : [...source.value];
    const rows = searched.filter((item) => filters.every((filter) => {
      const value = String(filter.value(item) ?? "");
      if (filter.type === "multi") {
        const selected = multiFilters.value[filter.key] || [];
        return !selected.length || selected.includes(value);
      }
      const expected = (textFilters.value[filter.key] || "").trim().toLocaleLowerCase();
      return !expected || value.toLocaleLowerCase().includes(expected);
    }));
    const option = sortOptions.find((candidate) => candidate.key === sortKey.value);
    if (!option) return rows;
    const direction = sortDirection.value === "asc" ? 1 : -1;
    return rows.sort((left, right) => {
      const a = option.value(left);
      const b = option.value(right);
      if (a == null || a === "") return b == null || b === "" ? 0 : 1;
      if (b == null || b === "") return -1;
      if (typeof a === "number" && typeof b === "number") return (a - b) * direction;
      return String(a).localeCompare(String(b), "zh-CN", { numeric: true, sensitivity: "base" }) * direction;
    });
  });
  const total = computed(() => arranged.value.length);
  const pageCount = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)));
  const visibleItems = computed(() => {
    const start = (page.value - 1) * pageSize.value;
    return arranged.value.slice(start, start + pageSize.value);
  });

  watch([query, sortKey, sortDirection, pageSize, textFilters, multiFilters], () => { page.value = 1; }, { deep: true });
  watch(pageCount, (count) => { if (page.value > count) page.value = count; });

  function clearFilters() {
    query.value = "";
    textFilters.value = {};
    multiFilters.value = {};
  }

  return { query, sortKey, sortDirection, page, pageSize, textFilters, multiFilters,
    filterOptions, activeFilterCount, total, pageCount, visibleItems, clearFilters };
}
