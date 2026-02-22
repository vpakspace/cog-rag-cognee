"""CLI ingestion script."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cog_rag_cognee.cognee_setup import apply_cognee_env  # noqa: E402
from cog_rag_cognee.config import get_settings  # noqa: E402
from cog_rag_cognee.service import PipelineService  # noqa: E402


async def main(file_paths: list[str]) -> None:
    settings = get_settings()
    apply_cognee_env(settings)

    svc = PipelineService()

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"SKIP: {fp} not found")
            continue

        print(f"Ingesting: {fp}")
        content = path.read_text(encoding="utf-8")
        await svc.add_text(content)
        print(f"  Added {len(content)} chars")

    print("Running cognify...")
    result = await svc.cognify()
    print(f"Cognify result: {result}")
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file1> [file2] ...")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
