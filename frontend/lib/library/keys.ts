// TanStack Query keys of the library. Plain TypeScript. Every library mutation invalidates
// the whole `["library"]` tree: the lists are small and a count in one list (the sidebar)
// changes when an item in another (the table) does.

export const LIBRARY_KEY = ["library"] as const;
export const collectionsKey = [...LIBRARY_KEY, "collections"] as const;
export const itemsKey = (collection: string | null) =>
  [...LIBRARY_KEY, "items", collection] as const;
