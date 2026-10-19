"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { InterestIn, InterestOut } from "@/lib/api/queries";
import { DESCRIPTION_MAX, NAME_MAX, parseInterest } from "@/lib/settings/form";

// Create or edit one interest. The text is checked here first (the same rules as the API),
// then sent; a name that already exists comes back from the server as a sentence.
export function InterestForm({
  initial,
  onSubmit,
  onCancel,
  pending,
  serverError,
}: {
  initial: InterestOut | null;
  onSubmit: (body: InterestIn) => void;
  onCancel?: () => void;
  pending: boolean;
  serverError: string | null;
}) {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [keywords, setKeywords] = useState(initial?.keywords.join(", ") ?? "");
  const [categories, setCategories] = useState(
    initial?.categories.join(", ") ?? "",
  );
  const [isActive, setIsActive] = useState(initial?.is_active ?? true);
  const [problems, setProblems] = useState<string[]>([]);
  const prefix = initial ? `interest-${initial.id}` : "new-interest";

  return (
    <form
      aria-label={initial ? `Edit ${initial.name}` : "New interest"}
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        const result = parseInterest({
          name,
          description,
          keywords,
          categories,
          isActive,
        });
        if (!result.ok) {
          setProblems(result.problems);
          return;
        }
        setProblems([]);
        onSubmit(result.body);
      }}
    >
      <div className="space-y-1">
        <label htmlFor={`${prefix}-name`} className="text-sm font-medium">
          Name
        </label>
        <Input
          id={`${prefix}-name`}
          value={name}
          maxLength={NAME_MAX}
          onChange={(e) => setName(e.target.value)}
          placeholder="Retrieval-augmented generation"
        />
      </div>
      <div className="space-y-1">
        <label
          htmlFor={`${prefix}-description`}
          className="text-sm font-medium"
        >
          Description
        </label>
        <textarea
          id={`${prefix}-description`}
          value={description}
          maxLength={DESCRIPTION_MAX}
          rows={3}
          onChange={(e) => setDescription(e.target.value)}
          className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring md:text-sm"
          placeholder="What you want to read about, in a sentence or two. It is embedded and compared with each paper's abstract."
        />
      </div>
      <div className="space-y-1">
        <label htmlFor={`${prefix}-keywords`} className="text-sm font-medium">
          Keywords
        </label>
        <Input
          id={`${prefix}-keywords`}
          value={keywords}
          onChange={(e) => setKeywords(e.target.value)}
          placeholder="rag, retrieval, reranking"
        />
        <p className="text-xs text-muted-foreground">
          Separated by commas. A keyword counts when the whole word or phrase is
          in a paper&apos;s title or abstract.
        </p>
      </div>
      <div className="space-y-1">
        <label htmlFor={`${prefix}-categories`} className="text-sm font-medium">
          arXiv categories
        </label>
        <Input
          id={`${prefix}-categories`}
          value={categories}
          onChange={(e) => setCategories(e.target.value)}
          placeholder="cs.CL, cs.IR"
        />
        <p className="text-xs text-muted-foreground">
          Optional. A paper in one of these gets a small boost; leave empty for
          any category.
        </p>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={isActive}
          onChange={(e) => setIsActive(e.target.checked)}
        />
        Active (used to rank the digest)
      </label>
      {problems.length > 0 && (
        <ul
          role="alert"
          className="list-disc space-y-1 pl-5 text-sm text-destructive"
        >
          {problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}
      {serverError && (
        <p role="alert" className="text-sm text-destructive">
          {serverError}
        </p>
      )}
      <div className="flex gap-2">
        <Button type="submit" size="sm" disabled={pending}>
          {initial ? "Save changes" : "Add interest"}
        </Button>
        {onCancel && (
          <Button type="button" size="sm" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}
