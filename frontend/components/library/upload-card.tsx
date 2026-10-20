"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { uploadPdf } from "@/lib/api/queries";
import { LIBRARY_KEY } from "@/lib/library/keys";
import {
  UPLOAD_COST_NOTE,
  formatBytes,
  titleFromFileName,
  uploadProblem,
} from "@/lib/library/upload";

// Upload a PDF that is not on arXiv (a journal paper). PAID: the PDF is parsed on the server
// and its chunks are embedded. Choosing a file or typing sends nothing; only the Upload
// button does, and it is off unless the file and the title are fine. The PDF is never stored
// or shown again, only the text read from it.
export function UploadCard({ collection }: { collection: string }) {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [year, setYear] = useState("");
  const upload = useMutation({
    mutationFn: () =>
      uploadPdf(
        {
          title: title.trim(),
          collection,
          year: /^\d{4}$/.test(year.trim()) ? Number(year) : null,
        },
        file as File,
      ),
    onSuccess: async () => {
      setFile(null);
      setTitle("");
      setYear("");
      await queryClient.invalidateQueries({ queryKey: LIBRARY_KEY });
    },
  });
  const problem = uploadProblem(file, title);
  const yearProblem =
    year.trim() !== "" && !/^\d{4}$/.test(year.trim())
      ? "The year has four digits."
      : null;

  return (
    <section
      aria-label="Upload a PDF"
      className="space-y-3 rounded-md border p-4"
    >
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">Upload a PDF</h3>
        <p className="text-xs text-muted-foreground">
          For a journal paper that is not on arXiv. The text is read on the
          server and indexed like any paper; the PDF itself is not kept. Cost:{" "}
          {UPLOAD_COST_NOTE}. Nothing is sent until you press Upload.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-[1fr_6rem]">
        <div className="space-y-1">
          <label htmlFor="upload-file" className="text-sm font-medium">
            PDF file
          </label>
          <Input
            id="upload-file"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(event) => {
              const picked = event.target.files?.[0] ?? null;
              setFile(picked);
              if (picked && title.trim() === "")
                setTitle(titleFromFileName(picked.name));
              upload.reset();
            }}
          />
          {file && (
            <p
              className="text-xs text-muted-foreground"
              data-testid="file-size"
            >
              {file.name}, {formatBytes(file.size)}
            </p>
          )}
        </div>
        <div className="space-y-1">
          <label htmlFor="upload-year" className="text-sm font-medium">
            Year
          </label>
          <Input
            id="upload-year"
            inputMode="numeric"
            value={year}
            maxLength={4}
            placeholder="optional"
            onChange={(event) => setYear(event.target.value)}
          />
        </div>
      </div>
      <div className="space-y-1">
        <label htmlFor="upload-title" className="text-sm font-medium">
          Title
        </label>
        <Input
          id="upload-title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button
          type="button"
          disabled={
            problem !== null || yearProblem !== null || upload.isPending
          }
          onClick={() => upload.mutate()}
        >
          Upload
        </Button>
        <p className="text-xs text-muted-foreground">
          {yearProblem ?? problem ?? `Adds it to ${collection}.`}
        </p>
      </div>
      {upload.isPending && (
        <p role="status" className="text-sm text-muted-foreground">
          Reading and indexing the PDF&hellip; this can take a minute.
        </p>
      )}
      {upload.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-2 text-sm text-destructive"
        >
          {upload.error.message}
        </p>
      )}
      {upload.data && (
        <p role="status" className="text-sm text-muted-foreground">
          {upload.data.created
            ? `Added "${upload.data.title}" (${upload.data.arxiv_id}): ${Object.values(upload.data.chunks).reduce((a, b) => a + b, 0)} passages indexed, ${upload.data.embedding_requests} embedding request${upload.data.embedding_requests === 1 ? "" : "s"}, about $${upload.data.cost_usd.toFixed(6)}.`
            : `"${upload.data.title}" was uploaded before. Nothing changed and nothing was charged.`}
        </p>
      )}
    </section>
  );
}
