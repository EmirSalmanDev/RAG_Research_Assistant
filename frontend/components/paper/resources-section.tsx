"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import type { PaperDetailsOut } from "@/lib/api/queries";
import { formatCount, formatDate } from "@/lib/format";
import { siteUrl } from "@/lib/links";

type Artifact = PaperDetailsOut["models"][number];

const SHOWN = 8; // artifacts listed per kind before "Show all"
const LINK =
  "font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

// The paper's code and Hugging Face artifacts (GET /papers/{id}/details). Every name
// comes from third-party data: it is shown as text, and only linked when its URL is an
// https page on github.com or huggingface.co.
export function ResourcesSection({ details }: { details: PaperDetailsOut }) {
  return (
    <section aria-labelledby="resources-heading" className="space-y-5">
      <h2 id="resources-heading" className="text-lg font-semibold">
        Code and models
      </h2>
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-muted-foreground">Code</h3>
        {details.repos.length === 0 ? (
          <p className="text-muted-foreground">
            No code repository is linked to this paper.
          </p>
        ) : (
          <ul className="space-y-2">
            {details.repos.map((repo) => {
              const url = siteUrl(repo.url, "https://github.com");
              const facts = [
                repo.stars != null ? `${formatCount(repo.stars)} stars` : null,
                repo.license,
                repo.last_push
                  ? `last push ${formatDate(repo.last_push)}`
                  : null,
              ].filter(Boolean);
              return (
                <li key={repo.full_name}>
                  {url ? (
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer nofollow"
                      className={LINK}
                    >
                      {repo.full_name}
                    </a>
                  ) : (
                    <span className="font-medium">{repo.full_name}</span>
                  )}
                  {facts.length > 0 && (
                    <span className="text-sm text-muted-foreground">
                      {" "}
                      ({facts.join(", ")})
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <ArtifactList title="Models" items={details.models} />
      <ArtifactList title="Datasets" items={details.datasets} />
      <ArtifactList title="Spaces" items={details.spaces} />
    </section>
  );
}

function ArtifactList({ title, items }: { title: string; items: Artifact[] }) {
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, SHOWN);
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-muted-foreground">
        {title} on Hugging Face{items.length > 0 ? ` (${items.length})` : ""}
      </h3>
      {items.length === 0 ? (
        <p className="text-muted-foreground">
          None linked to this paper on Hugging Face.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {shown.map((item) => {
            const url = siteUrl(item.url, "https://huggingface.co");
            const downloads = formatCount(item.downloads);
            const likes = formatCount(item.likes);
            return (
              <li
                key={item.repo_id}
                className="flex flex-wrap items-baseline gap-x-2"
              >
                {url ? (
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer nofollow"
                    className={`break-all ${LINK}`}
                  >
                    {item.repo_id}
                  </a>
                ) : (
                  <span className="break-all font-medium">{item.repo_id}</span>
                )}
                {item.pipeline_tag && (
                  <Badge variant="outline" className="font-normal">
                    {item.pipeline_tag}
                  </Badge>
                )}
                <span className="text-sm text-muted-foreground">
                  {[
                    downloads ? `${downloads} downloads` : null,
                    likes ? `${likes} likes` : null,
                  ]
                    .filter(Boolean)
                    .join(", ")}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      {items.length > SHOWN && (
        <button
          type="button"
          onClick={() => setAll(!all)}
          aria-expanded={all}
          className={`text-sm ${LINK}`}
        >
          {all ? "Show fewer" : `Show all ${items.length}`}
        </button>
      )}
    </div>
  );
}
