"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { fetchBibtex } from "@/lib/api/queries";
import { downloadText, copyText } from "./download";

// The collection's references as BibTeX, built from stored metadata on the server (no model,
// no network, free). Download saves the .bib file; Copy puts the text on the clipboard.
export function BibActions({
  collection,
  disabled,
}: {
  collection: string;
  disabled: boolean;
}) {
  const [copied, setCopied] = useState<"yes" | "failed" | null>(null);
  const download = useMutation({
    mutationFn: async () => {
      const bib = await fetchBibtex(collection);
      downloadText(
        bib.filename ?? "library.bib",
        bib.text,
        "text/plain;charset=utf-8",
      );
    },
  });
  const copy = useMutation({
    mutationFn: async () => {
      const bib = await fetchBibtex(collection);
      setCopied((await copyText(bib.text)) ? "yes" : "failed");
    },
    onMutate: () => setCopied(null),
  });
  const error = download.error ?? copy.error;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        type="button"
        variant="outline"
        disabled={disabled || download.isPending}
        onClick={() => download.mutate()}
      >
        Download .bib
      </Button>
      <Button
        type="button"
        variant="outline"
        disabled={disabled || copy.isPending}
        onClick={() => copy.mutate()}
      >
        Copy BibTeX
      </Button>
      <span role="status" className="text-sm text-muted-foreground">
        {copied === "yes" && "Copied to the clipboard."}
        {copied === "failed" && "Could not copy. Use the download instead."}
      </span>
      {error && (
        <span role="alert" className="text-sm text-destructive">
          {error.message}
        </span>
      )}
    </div>
  );
}
