"""PipelineService — thin wrapper over Cognee SDK."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import cognee
from cognee.modules.search.types.SearchType import SearchType

from cog_rag_cognee.config import get_settings
from cog_rag_cognee.docling_loader import DoclingLoader
from cog_rag_cognee.exceptions import IngestionError, SearchError
from cog_rag_cognee.models import QAResult, SearchResult

T = TypeVar("T")
logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)


async def retry_transient(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 2,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Call *func* with exponential backoff on transient errors.

    Only retries ConnectionError, TimeoutError, OSError.
    Non-transient errors (ValueError, etc.) are raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


_docling_loader: DoclingLoader | None = None


def _get_docling_loader() -> DoclingLoader:
    """Return a cached DoclingLoader singleton (GPU models are ~2 GB)."""
    global _docling_loader
    if _docling_loader is None:
        settings = get_settings()
        _docling_loader = DoclingLoader(use_gpu=settings.docling_use_gpu)
    return _docling_loader


class PipelineService:
    """Orchestrates Cognee SDK operations."""

    async def add_text(self, text: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add text data to Cognee."""
        try:
            await retry_transient(cognee.add, text, dataset_name=dataset_name)
        except Exception as exc:
            raise IngestionError(f"Failed to add text: {exc}") from exc
        logger.info("Added text (%d chars) to dataset '%s'", len(text), dataset_name)
        return {"status": "added", "chars": len(text), "dataset": dataset_name}

    async def add_file(self, file_path: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add file content to Cognee via DoclingLoader."""
        try:
            loader = _get_docling_loader()
            result = loader.load(file_path)
            await cognee.add(result.markdown, dataset_name=dataset_name)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Failed to add file '{file_path}': {exc}") from exc
        logger.info("Added file '%s' (%d chars)", file_path, len(result.markdown))
        return {
            "status": "added",
            "file": str(Path(file_path).name),
            "chars": len(result.markdown),
        }

    async def add_bytes(
        self, data: bytes, filename: str, dataset_name: str = "main"
    ) -> dict[str, Any]:
        """Add uploaded file bytes to Cognee via DoclingLoader."""
        try:
            loader = _get_docling_loader()
            result = loader.load_bytes(data, filename)
            await cognee.add(result.markdown, dataset_name=dataset_name)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Failed to add bytes '{filename}': {exc}") from exc
        logger.info("Added bytes '%s' (%d chars)", filename, len(result.markdown))
        return {"status": "added", "file": filename, "chars": len(result.markdown)}

    async def cognify(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Run Cognee ECL pipeline: extract entities, build graph, embed."""
        try:
            kwargs: dict[str, Any] = {}
            if dataset_name:
                kwargs["datasets"] = [dataset_name]
            result = await retry_transient(cognee.cognify, **kwargs)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Cognify failed: {exc}") from exc
        logger.info("Cognify completed: %s", result)
        return result

    async def search(
        self,
        query: str,
        search_type: str = "CHUNKS",
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search Cognee knowledge graph."""
        try:
            st = SearchType(search_type) if isinstance(search_type, str) else search_type
            raw_results = await retry_transient(
                cognee.search, query, query_type=st, top_k=limit
            )
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(f"Search failed: {exc}") from exc
        results = []
        for r in raw_results:
            content, score = self._extract_result(r)
            results.append(SearchResult(content=content, score=score))
        return results

    @staticmethod
    def _extract_result(r: Any) -> tuple[str, float]:
        """Extract text content and score from a Cognee search result.

        Cognee v0.5.2 returns heterogeneous types depending on SearchType:
        - CHUNKS/SUMMARIES: list[dict] with 'text' key
        - RAG_COMPLETION: list[str]
        - Object with .content / .relevance_score attributes
        """
        # v0.5.2: result is a list (of dicts or strings)
        if isinstance(r, list):
            parts = []
            for item in r:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else "", 0.5

        # Older SDK: object with .content attribute
        if hasattr(r, "content"):
            raw = r.content
            if isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)
            else:
                content = str(raw)
            score = r.relevance_score if hasattr(r, "relevance_score") else 0.5
            return content, max(0.0, min(1.0, float(score)))

        return str(r), 0.5

    async def query(
        self,
        question: str,
        search_type: str = "CHUNKS",
        limit: int = 5,
    ) -> QAResult:
        """Full RAG pipeline: search + format answer."""
        sources = await self.search(question, search_type=search_type, limit=limit)
        if sources:
            answer = sources[0].content
            confidence = sources[0].score
        else:
            answer = "No relevant information found."
            confidence = 0.0
        return QAResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            mode=search_type,
        )

    async def list_datasets(self) -> list[str]:
        """Return names of available Cognee datasets."""
        try:
            datasets = await cognee.datasets.list_datasets()
            return [ds.name for ds in datasets] if datasets else []
        except Exception as exc:
            logger.warning("Failed to list datasets: %s", exc)
            return []

    async def reset(self) -> None:
        """Reset all Cognee data."""
        try:
            await cognee.prune.prune_data()
        except Exception as exc:
            raise IngestionError(f"Data reset failed: {exc}") from exc
        logger.info("Data reset completed")

    async def reset_system(self) -> None:
        """Full system reset including metadata."""
        try:
            await cognee.prune.prune_system(metadata=True)
        except Exception as exc:
            raise IngestionError(f"System reset failed: {exc}") from exc
        logger.info("System reset completed")
