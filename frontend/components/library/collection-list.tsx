"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  createCollection,
  deleteCollection,
  updateCollection,
} from "@/lib/api/queries";
import type { CollectionOut } from "@/lib/api/queries";
import { LIBRARY_KEY } from "@/lib/library/keys";
import { nameProblem } from "@/lib/library/forms";
import { paperCount } from "@/lib/library/model";
import { cn } from "@/lib/utils";

// The collections: pick one, create one (also an empty one), rename or delete the picked
// one. Deleting asks first and removes the collection's items, never the papers.
export function CollectionList({
  collections,
  selected,
  onSelect,
}: {
  collections: readonly CollectionOut[];
  selected: string | null;
  onSelect: (name: string | null) => void;
}) {
  const queryClient = useQueryClient();
  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: LIBRARY_KEY });
  const [newName, setNewName] = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameTo, setRenameTo] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const current = collections.find((c) => c.name === selected) ?? null;

  const create = useMutation({
    mutationFn: (name: string) => createCollection({ name }),
    onSuccess: async (created) => {
      setNewName("");
      await refresh();
      onSelect(created.name);
    },
  });
  const rename = useMutation({
    mutationFn: (vars: { from: string; to: string }) =>
      updateCollection(vars.from, { name: vars.to }),
    onSuccess: async (renamed) => {
      setRenaming(null);
      await refresh();
      onSelect(renamed.name);
    },
  });
  const remove = useMutation({
    mutationFn: (name: string) => deleteCollection(name),
    onSuccess: async () => {
      setConfirmDelete(false);
      onSelect(null);
      await refresh();
    },
  });

  const createProblem = newName === "" ? "" : nameProblem(newName);
  const renameProblem = renaming === null ? null : nameProblem(renameTo);
  const error = create.error ?? rename.error ?? remove.error;

  return (
    <section aria-label="Collections" className="space-y-4">
      <h2 className="text-lg font-semibold tracking-tight">Collections</h2>
      {collections.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No collection yet. Create one below, or press Save on a digest card:
          saved papers go to a collection called &ldquo;Saved&rdquo;.
        </p>
      ) : (
        <ul className="space-y-1">
          {collections.map((c) => (
            <li key={c.name}>
              <button
                type="button"
                aria-current={c.name === selected ? "true" : undefined}
                onClick={() => {
                  onSelect(c.name);
                  setConfirmDelete(false);
                  setRenaming(null);
                }}
                className={cn(
                  "flex w-full items-baseline justify-between gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  c.name === selected
                    ? "border-primary bg-accent font-medium text-accent-foreground"
                    : "bg-card hover:bg-accent",
                )}
              >
                <span className="truncate">{c.name}</span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {paperCount(c.paper_count)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <form
        className="space-y-1"
        onSubmit={(event) => {
          event.preventDefault();
          if (nameProblem(newName) === null) create.mutate(newName);
        }}
      >
        <label htmlFor="new-collection" className="text-sm font-medium">
          New collection
        </label>
        <div className="flex gap-2">
          <Input
            id="new-collection"
            value={newName}
            maxLength={150}
            placeholder="Thesis, chapter 2"
            onChange={(event) => setNewName(event.target.value)}
          />
          <Button
            type="submit"
            variant="outline"
            disabled={nameProblem(newName) !== null || create.isPending}
          >
            Create
          </Button>
        </div>
        {createProblem && (
          <p className="text-xs text-destructive">{createProblem}</p>
        )}
      </form>

      {current && (
        <div className="space-y-2 border-t pt-4">
          {renaming === null ? (
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  setRenaming(current.name);
                  setRenameTo(current.name);
                  setConfirmDelete(false);
                }}
              >
                Rename
              </Button>
              {!confirmDelete && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmDelete(true)}
                >
                  Delete collection
                </Button>
              )}
            </div>
          ) : (
            <form
              className="space-y-1"
              onSubmit={(event) => {
                event.preventDefault();
                if (renameProblem === null)
                  rename.mutate({ from: renaming, to: renameTo });
              }}
            >
              <label
                htmlFor="rename-collection"
                className="text-sm font-medium"
              >
                Rename &ldquo;{renaming}&rdquo; to
              </label>
              <div className="flex gap-2">
                <Input
                  id="rename-collection"
                  value={renameTo}
                  maxLength={150}
                  onChange={(event) => setRenameTo(event.target.value)}
                />
                <Button
                  type="submit"
                  size="sm"
                  disabled={renameProblem !== null || rename.isPending}
                >
                  Save name
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setRenaming(null)}
                >
                  Cancel
                </Button>
              </div>
              {renameProblem && (
                <p className="text-xs text-destructive">{renameProblem}</p>
              )}
            </form>
          )}
          {confirmDelete && (
            <div
              role="alertdialog"
              aria-label="Confirm delete"
              className="space-y-2 rounded-md border border-destructive/40 p-3 text-sm"
            >
              <p>
                Delete &ldquo;{current.name}&rdquo; and its{" "}
                {paperCount(current.paper_count)} from the library? The papers
                themselves stay in the database.
              </p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  disabled={remove.isPending}
                  onClick={() => remove.mutate(current.name)}
                >
                  Yes, delete it
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setConfirmDelete(false)}
                >
                  Keep it
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
      {error && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-2 text-sm text-destructive"
        >
          {error.message}
        </p>
      )}
    </section>
  );
}
