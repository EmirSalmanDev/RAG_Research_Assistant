"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchSettings, saveSettings } from "@/lib/api/queries";
import type { SettingsOut } from "@/lib/api/queries";
import {
  BUDGET_MAX,
  DIGEST_SIZE_RANGE,
  TOP_N_RANGE,
  formatUsd,
  parseSettings,
} from "@/lib/settings/form";

// Digest size, the arXiv categories the daily fetch uses, how many papers get full text
// each night, and the monthly budget cap. The cap is stored and shown next to this month's
// spending, but nothing is paused when it is reached yet.
export function DigestSettings() {
  const settings = useQuery({ queryKey: ["settings"], queryFn: fetchSettings });
  if (settings.isPending) {
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading settings&hellip;
      </p>
    );
  }
  if (settings.isError) {
    return (
      <p role="alert" className="text-sm text-destructive">
        Settings could not be loaded: {settings.error.message}
      </p>
    );
  }
  return <SettingsForm settings={settings.data} />;
}

function SettingsForm({ settings }: { settings: SettingsOut }) {
  const queryClient = useQueryClient();
  const [digestSize, setDigestSize] = useState(String(settings.digest_size));
  const [topN, setTopN] = useState(String(settings.fulltext_top_n));
  const [categories, setCategories] = useState(
    settings.categories_overridden ? settings.categories.join(", ") : "",
  );
  const [budget, setBudget] = useState(
    settings.budget_cap_usd === null ? "" : String(settings.budget_cap_usd),
  );
  const [problems, setProblems] = useState<string[]>([]);
  const save = useMutation({
    mutationFn: saveSettings,
    onSuccess: (saved) => queryClient.setQueryData(["settings"], saved),
  });

  return (
    <form
      aria-label="Digest settings"
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        save.reset();
        const result = parseSettings({
          digestSize,
          fulltextTopN: topN,
          categories,
          budgetCap: budget,
        });
        if (!result.ok) {
          setProblems(result.problems);
          return;
        }
        setProblems([]);
        save.mutate(result.body);
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <label htmlFor="digest-size" className="text-sm font-medium">
            Digest size
          </label>
          <Input
            id="digest-size"
            inputMode="numeric"
            value={digestSize}
            onChange={(e) => setDigestSize(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Papers in each morning&apos;s list ({DIGEST_SIZE_RANGE[0]} to{" "}
            {DIGEST_SIZE_RANGE[1]}).
          </p>
        </div>
        <div className="space-y-1">
          <label htmlFor="fulltext-top-n" className="text-sm font-medium">
            Full-text papers per night
          </label>
          <Input
            id="fulltext-top-n"
            inputMode="numeric"
            value={topN}
            onChange={(e) => setTopN(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            The best-ranked papers also get their full text indexed (
            {TOP_N_RANGE[0]} to {TOP_N_RANGE[1]}; 0 means only saved and
            requested papers).
          </p>
        </div>
      </div>
      <div className="space-y-1">
        <label htmlFor="settings-categories" className="text-sm font-medium">
          arXiv categories to fetch
        </label>
        <Input
          id="settings-categories"
          value={categories}
          onChange={(e) => setCategories(e.target.value)}
          placeholder={settings.categories_default.join(", ")}
        />
        <p className="text-xs text-muted-foreground">
          Separated by commas; empty uses the default list (
          {settings.categories_default.join(", ")}). A category you add is
          fetched from the next daily run on; papers it already has are not
          backfilled.
        </p>
      </div>
      <div className="space-y-1">
        <label htmlFor="budget-cap" className="text-sm font-medium">
          Monthly budget cap (USD)
        </label>
        <Input
          id="budget-cap"
          inputMode="decimal"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
          placeholder={`no cap (up to ${BUDGET_MAX})`}
        />
        <p className="text-xs text-muted-foreground">
          Spent this month:{" "}
          <strong data-testid="month-spend">
            {formatUsd(settings.month_to_date_usd)}
          </strong>
          . The cap is stored and shown here only; nothing is paused when it is
          reached yet.
        </p>
      </div>
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
      {save.isError && (
        <p role="alert" className="text-sm text-destructive">
          Settings were not saved: {save.error.message}
        </p>
      )}
      {save.isSuccess && (
        <p role="status" className="text-sm text-muted-foreground">
          Settings saved.
        </p>
      )}
      <Button type="submit" size="sm" disabled={save.isPending}>
        Save settings
      </Button>
    </form>
  );
}
