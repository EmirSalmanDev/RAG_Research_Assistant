"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { InterestForm } from "@/components/settings/interest-form";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingState } from "@/components/ui/loading-state";
import {
  createInterest,
  deleteInterest,
  fetchInterests,
  updateInterest,
} from "@/lib/api/queries";
import type { InterestIn, InterestOut } from "@/lib/api/queries";

const KEY = ["interests"] as const;

// The interests the digest ranks against. Saving one costs nothing: its embedding is made
// by the nightly run (or `make digest`), which is why a new or edited interest says so.
export function InterestsSection() {
  const queryClient = useQueryClient();
  const interests = useQuery({ queryKey: KEY, queryFn: fetchInterests });
  const [editing, setEditing] = useState<number | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);

  // The digest page shows an "add interests" note while there are none, so it is read
  // again after any change here.
  const refresh = () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: KEY }),
      queryClient.invalidateQueries({ queryKey: ["digest"] }),
    ]);
  // The "add" form is emptied after a successful add (a new key remounts it), but not
  // after a failed one, so a duplicate name can be fixed without retyping everything.
  const [addFormKey, setAddFormKey] = useState(0);
  const create = useMutation({
    mutationFn: (body: InterestIn) => createInterest(body),
    onSuccess: async () => {
      setAddFormKey((k) => k + 1);
      await refresh();
    },
  });
  const update = useMutation({
    mutationFn: ({ id, body }: { id: number; body: InterestIn }) =>
      updateInterest(id, body),
    onSuccess: async () => {
      setEditing(null);
      await refresh();
    },
  });
  const remove = useMutation({
    mutationFn: (id: number) => deleteInterest(id),
    onSuccess: async () => {
      setConfirming(null);
      await refresh();
    },
  });

  return (
    <section aria-labelledby="interests-heading" className="space-y-4">
      <div>
        <h2 id="interests-heading" className="text-xl font-semibold">
          Interests
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          The digest ranks each new paper against these. Save and Dismiss on the
          digest nudge an interest&apos;s weight up or down. A new or edited
          interest is embedded by the next nightly run, so it affects the
          ranking from then on.
        </p>
      </div>

      {interests.isPending && (
        <LoadingState label="Loading interests…" lines={2} />
      )}
      {interests.isError && (
        <ErrorState
          what="Interests"
          error={interests.error}
          onRetry={() => void interests.refetch()}
        />
      )}
      {interests.data?.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No interests yet. Add one below.
        </p>
      )}
      <ul className="space-y-3">
        {interests.data?.map((interest) => (
          <li key={interest.id} className="rounded-lg border bg-card p-4">
            {editing === interest.id ? (
              <InterestForm
                initial={interest}
                pending={update.isPending}
                serverError={update.isError ? update.error.message : null}
                onCancel={() => {
                  update.reset();
                  setEditing(null);
                }}
                onSubmit={(body) => update.mutate({ id: interest.id, body })}
              />
            ) : (
              <InterestRow
                interest={interest}
                confirming={confirming === interest.id}
                deleting={remove.isPending}
                deleteError={
                  remove.isError && confirming === interest.id
                    ? remove.error.message
                    : null
                }
                onEdit={() => {
                  update.reset();
                  setEditing(interest.id);
                }}
                onAskDelete={() => {
                  remove.reset();
                  setConfirming(interest.id);
                }}
                onCancelDelete={() => setConfirming(null)}
                onDelete={() => remove.mutate(interest.id)}
              />
            )}
          </li>
        ))}
      </ul>

      <div className="rounded-lg border border-dashed p-4">
        <h3 className="mb-3 font-medium">Add an interest</h3>
        <InterestForm
          key={addFormKey}
          initial={null}
          pending={create.isPending}
          serverError={create.isError ? create.error.message : null}
          onSubmit={(body) => create.mutate(body)}
        />
      </div>
    </section>
  );
}

function InterestRow({
  interest,
  confirming,
  deleting,
  deleteError,
  onEdit,
  onAskDelete,
  onCancelDelete,
  onDelete,
}: {
  interest: InterestOut;
  confirming: boolean;
  deleting: boolean;
  deleteError: string | null;
  onEdit: () => void;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-medium">{interest.name}</h3>
        {!interest.is_active && <Badge variant="secondary">Inactive</Badge>}
        <span
          className="text-xs text-muted-foreground"
          title="Moved by Save (+0.10) and Dismiss (-0.05) on the digest, between 0.5 and 2.0"
        >
          weight {interest.weight.toFixed(2)}
        </span>
        {interest.is_active && !interest.embedded && (
          <span className="text-xs text-muted-foreground">
            waiting for the next nightly embedding
          </span>
        )}
      </div>
      {interest.description && (
        <p className="text-sm text-muted-foreground">{interest.description}</p>
      )}
      {(interest.keywords.length > 0 || interest.categories.length > 0) && (
        <ul
          aria-label={`Keywords and categories of ${interest.name}`}
          className="flex flex-wrap gap-1.5"
        >
          {interest.keywords.map((k) => (
            <li key={`k-${k}`}>
              <Badge variant="outline" className="font-normal">
                {k}
              </Badge>
            </li>
          ))}
          {interest.categories.map((c) => (
            <li key={`c-${c}`}>
              <Badge variant="secondary" className="font-normal">
                {c}
              </Badge>
            </li>
          ))}
        </ul>
      )}
      <div className="flex items-center gap-2 pt-1">
        {confirming ? (
          <>
            <span className="text-sm">Delete this interest?</span>
            <Button
              size="sm"
              variant="destructive"
              disabled={deleting}
              onClick={onDelete}
              aria-label={`Confirm delete ${interest.name}`}
            >
              Delete
            </Button>
            <Button size="sm" variant="ghost" onClick={onCancelDelete}>
              Keep
            </Button>
          </>
        ) : (
          <>
            <Button
              size="sm"
              variant="outline"
              onClick={onEdit}
              aria-label={`Edit ${interest.name}`}
            >
              Edit
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={onAskDelete}
              aria-label={`Delete ${interest.name}`}
            >
              Delete
            </Button>
          </>
        )}
      </div>
      {deleteError && (
        <p role="alert" className="text-sm text-destructive">
          {deleteError}
        </p>
      )}
    </div>
  );
}
