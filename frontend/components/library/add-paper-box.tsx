"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { addItem } from "@/lib/api/queries";
import { referenceProblem } from "@/lib/library/forms";
import { LIBRARY_KEY } from "@/lib/library/keys";

// Add a paper that is already in the database, by arXiv id or arxiv.org link. A paper that
// is not in the database is refused with a hint (adding it needs the worker, a later step).
export function AddPaperBox({ collection }: { collection: string }) {
  const queryClient = useQueryClient();
  const [text, setText] = useState("");
  const [added, setAdded] = useState<string | null>(null);
  const add = useMutation({
    mutationFn: (reference: string) =>
      addItem({ reference, collection, status: "to_read" }),
    onSuccess: async (item) => {
      setText("");
      setAdded(item.title);
      await queryClient.invalidateQueries({ queryKey: LIBRARY_KEY });
    },
    onMutate: () => setAdded(null),
  });
  const problem = referenceProblem(text);
  return (
    <form
      className="space-y-1"
      onSubmit={(event) => {
        event.preventDefault();
        if (problem === null) add.mutate(text.trim());
      }}
    >
      <label htmlFor="add-paper" className="text-sm font-medium">
        Add a paper by arXiv id or link
      </label>
      <div className="flex gap-2">
        <Input
          id="add-paper"
          value={text}
          placeholder="1706.03762 or https://arxiv.org/abs/1706.03762"
          onChange={(event) => setText(event.target.value)}
        />
        <Button
          type="submit"
          variant="outline"
          disabled={problem !== null || add.isPending}
        >
          Add paper
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Only papers already in the database can be added. Nothing is downloaded
        and nothing costs money.
      </p>
      {add.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-2 text-sm text-destructive"
        >
          {add.error.message}
        </p>
      )}
      {added && (
        <p role="status" className="text-sm text-muted-foreground">
          Added &ldquo;{added}&rdquo;.
        </p>
      )}
    </form>
  );
}
