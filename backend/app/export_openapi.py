"""Write the API's OpenAPI schema to a file, straight from the app code.

    python -m app.export_openapi ../frontend/openapi.json          # write it
    python -m app.export_openapi ../frontend/openapi.json --check  # exit 1 if it is stale

The frontend generates its TypeScript client from this file, so the two cannot drift.
It builds the schema from the route definitions only: no server, no stores, no network,
no API keys. `--check` is for `make lint`: it fails when a route changed and the file
was not regenerated.
"""

import argparse
import json
import sys
from pathlib import Path

from app.main import app


def render() -> str:
    # Sorted keys and a final newline: the same app always gives the same bytes.
    return json.dumps(app.openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if stale")
    args = parser.parse_args(argv)

    text = render()
    if args.check:
        current = args.path.read_text(encoding="utf-8") if args.path.exists() else None
        if current != text:
            print(
                f"{args.path} is out of date: run `make openapi` and commit the result",
                file=sys.stderr,
            )
            return 1
        return 0
    args.path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
