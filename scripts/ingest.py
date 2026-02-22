"""CLI ingestion script with Docling document loader support."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cog_rag_cognee.cognee_setup import apply_cognee_env  # noqa: E402
from cog_rag_cognee.config import get_settings  # noqa: E402
from cog_rag_cognee.service import PipelineService  # noqa: E402


async def main(file_paths: list[str], use_gpu: bool = False) -> None:
    settings = get_settings()

    if use_gpu:
        os.environ["DOCLING_USE_GPU"] = "true"

    apply_cognee_env(settings)

    svc = PipelineService()

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"SKIP: {fp} not found")
            continue

        print(f"Ingesting: {fp}")
        try:
            result = await svc.add_file(str(path))
            print(f"  Added {result['chars']} chars")
        except ImportError as exc:
            print(f"  ERROR: {exc}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("Running cognify...")
    result = await svc.cognify()
    print(f"Cognify result: {result}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Cognee")
    parser.add_argument("files", nargs="+", help="File paths to ingest")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU acceleration for Docling"
    )
    args = parser.parse_args()
    asyncio.run(main(args.files, use_gpu=args.use_gpu))
