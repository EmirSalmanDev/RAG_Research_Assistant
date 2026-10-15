"""Abstract and section chunks must not change when other chunk kinds are added.

The digest below covers every markdown fixture: each chunk's id (= its Qdrant point id),
ord, type, section path, text, token count and embedding text. It was computed with the
chunking code from before analysis chunks were added (commit 0df5651) and is unchanged
since. If this test fails, existing points get new ids or new vectors: that needs a
re-index of the corpus and a retrieval evaluation, never just a new digest.
"""

import hashlib
import json
from pathlib import Path

from app.ingest.chunk import abstract_chunk, chunk_paper
from app.ingest.clean import clean_markdown

FIXTURES = Path(__file__).parents[1] / "fixtures" / "markdown"
EXPECTED_CHUNKS = 31
EXPECTED_DIGEST = "2d877e1c4f8cbd64fefcacdfa0318a8f82d30b97179ac50c430b183d02e8aad2"


def test_abstract_and_section_chunks_are_unchanged() -> None:
    digest, n = hashlib.sha256(), 0
    for path in sorted(FIXTURES.glob("*.md")):
        arxiv_id, title = f"2601.{n:05d}", f"Invented {path.stem}"
        chunks = [abstract_chunk(arxiv_id, 3, title, f"An invented abstract for {path.stem}.")]
        chunks += chunk_paper(arxiv_id, 3, title, clean_markdown(path.read_text()).markdown)
        for c in chunks:
            row = [str(c.id), c.ord, c.chunk_type, c.section_path, c.text, c.token_count,
                   c.embedding_text]  # fmt: skip
            digest.update(json.dumps(row).encode())
            n += 1
    assert (n, digest.hexdigest()) == (EXPECTED_CHUNKS, EXPECTED_DIGEST)
