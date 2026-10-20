"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { removeItem, updateItem } from "@/lib/api/queries";
import type { CollectionOut, ItemOut } from "@/lib/api/queries";
import { formatDate } from "@/lib/format";
import { LIBRARY_KEY } from "@/lib/library/keys";
import { NOTES_MAX_CHARS } from "@/lib/library/forms";
import { ITEM_EVIDENCE_LABEL, STATUSES } from "@/lib/library/model";
import type { LibraryStatus } from "@/lib/library/model";
import { arxivAbsUrl, paperPath } from "@/lib/links";
import { cn } from "@/lib/utils";

const SELECT_CLASS =
  "h-8 rounded-md border border-input bg-background px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

function NoteCell({ item }: { item: ItemOut }) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(item.notes ?? "");
  const save = useMutation({
    mutationFn: (notes: string | null) =>
      updateItem(item.arxiv_id, item.collection, { notes }),
    onSuccess: async () => {
      setEditing(false);
      await queryClient.invalidateQueries({ queryKey: LIBRARY_KEY });
    },
  });
  if (!editing) {
    return (
      <div className="space-y-1">
        {item.notes ? (
          <p className="whitespace-pre-wrap text-sm">{item.notes}</p>
        ) : null}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-label={`${item.notes ? "Edit" : "Add"} note for ${item.title}`}
          onClick={() => {
            setDraft(item.notes ?? "");
            setEditing(true);
          }}
        >
          {item.notes ? "Edit note" : "Add note"}
        </Button>
      </div>
    );
  }
  return (
    <form
      className="space-y-1"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate(draft.trim() === "" ? null : draft);
      }}
    >
      <textarea
        aria-label={`Note for ${item.title}`}
        value={draft}
        maxLength={NOTES_MAX_CHARS}
        rows={3}
        onChange={(event) => setDraft(event.target.value)}
        className="w-full rounded-md border border-input bg-transparent px-2 py-1 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      />
      <div className="flex gap-2">
        <Button type="submit" size="sm" disabled={save.isPending}>
          Save note
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setEditing(false)}
        >
          Cancel
        </Button>
      </div>
      {save.isError && (
        <p role="alert" className="text-xs text-destructive">
          {save.error.message}
        </p>
      )}
    </form>
  );
}

function ItemRow({
  item,
  others,
}: {
  item: ItemOut;
  others: readonly CollectionOut[];
}) {
  const queryClient = useQueryClient();
  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: LIBRARY_KEY });
  const setStatus = useMutation({
    mutationFn: (status: LibraryStatus) =>
      updateItem(item.arxiv_id, item.collection, { status }),
    onSuccess: refresh,
  });
  const move = useMutation({
    mutationFn: (collection: string) =>
      updateItem(item.arxiv_id, item.collection, { collection }),
    onSuccess: refresh,
  });
  const remove = useMutation({
    mutationFn: () => removeItem(item.arxiv_id, item.collection),
    onSuccess: refresh,
  });
  const abs = arxivAbsUrl(item.abs_url);
  const error = setStatus.error ?? move.error ?? remove.error;

  return (
    <tr className="border-t align-top">
      <td className="max-w-sm space-y-1 p-3">
        <Link
          href={paperPath(item.arxiv_id)}
          className="font-medium leading-snug hover:underline"
        >
          {item.title}
        </Link>
        <p className="text-xs text-muted-foreground">
          {item.authors.slice(0, 3).join(", ") || "Unknown authors"}
          {item.authors.length > 3 ? " et al." : ""},{" "}
          {formatDate(item.published_at)}
        </p>
        <p className="text-xs text-muted-foreground">
          {abs ? (
            <a
              href={abs}
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              arXiv {item.arxiv_id}
            </a>
          ) : (
            "uploaded document"
          )}
        </p>
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error.message}
          </p>
        )}
      </td>
      <td className="p-3">
        <Badge
          variant={item.evidence === "full_text" ? "secondary" : "outline"}
          className={cn(item.evidence !== "full_text" && "border-dashed")}
        >
          {ITEM_EVIDENCE_LABEL[item.evidence]}
        </Badge>
      </td>
      <td className="p-3">
        <select
          aria-label={`Status of ${item.title}`}
          className={SELECT_CLASS}
          value={item.status}
          disabled={setStatus.isPending}
          onChange={(event) =>
            setStatus.mutate(event.target.value as LibraryStatus)
          }
        >
          {STATUSES.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </td>
      <td className="min-w-48 p-3">
        <NoteCell key={item.notes ?? ""} item={item} />
      </td>
      <td className="space-y-2 p-3">
        {others.length > 0 && (
          <select
            aria-label={`Move ${item.title} to`}
            className={SELECT_CLASS}
            value=""
            disabled={move.isPending}
            onChange={(event) => {
              if (event.target.value) move.mutate(event.target.value);
            }}
          >
            <option value="">Move to&hellip;</option>
            {others.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-label={`Remove ${item.title}`}
          disabled={remove.isPending}
          onClick={() => remove.mutate()}
        >
          Remove
        </Button>
      </td>
    </tr>
  );
}

// The papers of one collection. Status, notes, move and remove each change one item and
// nothing else; none of them costs anything.
export function ItemsTable({
  items,
  collections,
  collection,
}: {
  items: readonly ItemOut[];
  collections: readonly CollectionOut[];
  collection: string;
}) {
  const others = collections.filter((c) => c.name !== collection);
  if (items.length === 0) {
    return (
      <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
        This collection is empty. Add a paper below by its arXiv id or link
        (only papers already in the database), or upload a PDF.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">Papers in {collection}</caption>
        <thead className="bg-muted/50 text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="p-3 font-medium">
              Paper
            </th>
            <th scope="col" className="p-3 font-medium">
              Evidence
            </th>
            <th scope="col" className="p-3 font-medium">
              Status
            </th>
            <th scope="col" className="p-3 font-medium">
              Notes
            </th>
            <th scope="col" className="p-3 font-medium">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <ItemRow key={item.arxiv_id} item={item} others={others} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
