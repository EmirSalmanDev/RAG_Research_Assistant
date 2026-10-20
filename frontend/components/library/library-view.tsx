"use client";

import { useQuery } from "@tanstack/react-query";
import { useReducer, useState } from "react";

import { SourcePanel } from "@/components/chat/source-panel";
import { fetchCollections, fetchItems } from "@/lib/api/queries";
import { panelReducer } from "@/lib/chat/panel";
import { collectionsKey, itemsKey } from "@/lib/library/keys";
import { paperCount, sortCollections } from "@/lib/library/model";

import { AddPaperBox } from "./add-paper-box";
import { BibActions } from "./bib-actions";
import { CollectionList } from "./collection-list";
import { ItemsTable } from "./items-table";
import { ReviewPanel } from "./review-panel";
import { UploadCard } from "./upload-card";

export function LibraryView() {
  const [picked, setPicked] = useState<string | null>(null);
  const [panel, dispatchPanel] = useReducer(panelReducer, null);

  const collectionsQuery = useQuery({
    queryKey: collectionsKey,
    queryFn: fetchCollections,
  });
  const collections = sortCollections(collectionsQuery.data ?? []);
  // The picked collection, or the first one when none is picked (or it was deleted).
  const selected =
    collections.find((c) => c.name === picked)?.name ??
    collections[0]?.name ??
    null;
  const current = collections.find((c) => c.name === selected) ?? null;

  const itemsQuery = useQuery({
    queryKey: itemsKey(selected),
    queryFn: () => fetchItems(selected ?? ""),
    enabled: selected !== null,
  });

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8 px-4 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
        <p className="text-sm text-muted-foreground">
          Your collections of papers, with a reading status and notes, a BibTeX
          file, and a related-work draft built from them. Only the Upload and
          Generate review buttons cost money; everything else is free.
        </p>
      </div>

      {collectionsQuery.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          {collectionsQuery.error.message}
        </p>
      )}

      <div className="grid gap-8 lg:grid-cols-[18rem_1fr]">
        <CollectionList
          collections={collections}
          selected={selected}
          onSelect={setPicked}
        />

        <div className="min-w-0 space-y-8">
          {current === null ? (
            <p className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
              {collectionsQuery.isPending
                ? "Loading the library…"
                : "Create a collection to start."}
            </p>
          ) : (
            <>
              <section aria-label="Papers" className="space-y-4">
                <div className="space-y-1">
                  <h2 className="text-lg font-semibold tracking-tight">
                    {current.name}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {paperCount(current.paper_count)}
                    {current.description ? `. ${current.description}` : ""}
                  </p>
                </div>
                {itemsQuery.isError && (
                  <p role="alert" className="text-sm text-destructive">
                    {itemsQuery.error.message}
                  </p>
                )}
                {itemsQuery.data && (
                  <ItemsTable
                    items={itemsQuery.data}
                    collections={collections}
                    collection={current.name}
                  />
                )}
                <BibActions
                  collection={current.name}
                  disabled={current.paper_count === 0}
                />
                <AddPaperBox collection={current.name} />
                <UploadCard collection={current.name} />
              </section>
            </>
          )}

          <ReviewPanel
            collections={collections}
            initial={selected}
            activeChunkId={panel?.source.chunk_id ?? null}
            onOpen={(source, all) =>
              dispatchPanel({ type: "open", source, all })
            }
            onNewDraft={() => dispatchPanel({ type: "close" })}
          />
        </div>
      </div>

      <SourcePanel
        source={panel?.source ?? null}
        sources={panel?.all ?? []}
        onSelect={(source) => dispatchPanel({ type: "select", source })}
        onClose={() => dispatchPanel({ type: "close" })}
      />
    </div>
  );
}
