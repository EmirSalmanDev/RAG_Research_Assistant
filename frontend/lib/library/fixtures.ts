// An invented review for the library's unit tests (never real paper text).

import type { ReviewOut } from "./model.ts";

export const CHUNK_A = "00000000-0000-4000-8000-0000000000a1";
export const CHUNK_B = "00000000-0000-4000-8000-0000000000b2";
export const CHUNK_C = "00000000-0000-4000-8000-0000000000c3";

export const REVIEW: ReviewOut = {
  collection: "Thesis ch. 2: adapters",
  focus: "small adapters",
  sections: [
    {
      heading: "Adapters",
      paragraphs: [
        {
          sentences: [
            {
              text: "Several lines of work follow.",
              kind: "transition",
              source_numbers: [],
            },
            {
              text: "Example (2020) trains small adapters.",
              kind: "claim",
              source_numbers: [1],
            },
            {
              text: "Sample (2021) compares two methods.\n",
              kind: "claim",
              source_numbers: [1, 3],
            },
          ],
        },
      ],
    },
  ],
  references: [
    {
      n: 1,
      arxiv_id: "2601.00001",
      bibtex_key: "example2020invented",
      title: "Invented | Paper One",
      authors: ["Ada Example"],
      year: 2020,
      chunk_id: CHUNK_A,
      section_path: "Analysis > Summary",
      published_at: "2020-03-04T00:00:00+00:00",
      chunk_type: "analysis",
      evidence: "analysis",
      abs_url: "https://arxiv.org/abs/2601.00001",
      cited_count: 2,
    },
    {
      n: 2,
      arxiv_id: "2601.00002",
      bibtex_key: "sample2021unused",
      title: "Invented Paper Two",
      authors: ["Bo Sample"],
      year: 2021,
      chunk_id: CHUNK_B,
      section_path: "Abstract",
      published_at: "2021-03-04T00:00:00+00:00",
      chunk_type: "abstract",
      evidence: "abstract",
      abs_url: "https://arxiv.org/abs/2601.00002",
      cited_count: 0,
    },
    {
      n: 3,
      arxiv_id: "local-0123456789ab",
      bibtex_key: "anon2022uploaded",
      title: "An Uploaded Paper",
      authors: [],
      year: 2022,
      chunk_id: CHUNK_C,
      section_path: "Abstract",
      published_at: "2022-01-01T00:00:00+00:00",
      chunk_type: "abstract",
      evidence: "abstract",
      abs_url: null,
      cited_count: 1,
    },
  ],
  counts: {
    papers: 3,
    papers_without_analysis: 2,
    papers_omitted: 0,
    papers_uncitable: 0,
    claims: 2,
    transitions: 1,
    dropped_uncited: 1,
    dropped_transitions: 0,
    invalid_citations_dropped: 0,
    map_calls: 0,
    map_notes_dropped: 0,
  },
  tokens_in: 2400,
  tokens_out: 300,
  cost_usd: 0.00054,
  prompt_versions: { write: "review_write_v1" },
};
