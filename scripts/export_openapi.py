"""Export FastAPI OpenAPI schema to docs/site/openapi.json."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app


def main() -> None:
    output_path = REPO_ROOT / "docs" / "site" / "openapi.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = app.openapi()
    output_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"OpenAPI schema exported to: {output_path}")


if __name__ == "__main__":
    main()
