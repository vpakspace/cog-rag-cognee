"""PipelineService — thin wrapper over Cognee SDK."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cognee
from cognee.modules.search.types.SearchType import SearchType

from cog_rag_cognee.config import get_settings
from cog_rag_cognee.docling_loader import DoclingLoader
from cog_rag_cognee.models import QAResult, SearchResult

logger = logging.getLogger(__name__)


class PipelineService:
    """Orchestrates Cognee SDK operations."""

    async def add_text(self, text: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add text data to Cognee."""
        await cognee.add(text, dataset_name=dataset_name)
        logger.info("Added text (%d chars) to dataset '%s'", len(text), dataset_name)
        return {"status": "added", "chars": len(text), "dataset": dataset_name}

    async def add_file(self, file_path: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add file content to Cognee via DoclingLoader."""
        settings = get_settings()
        loader = DoclingLoader(use_gpu=settings.docling_use_gpu)
        result = loader.load(file_path)
        await cognee.add(result.markdown, dataset_name=dataset_name)
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
        settings = get_settings()
        loader = DoclingLoader(use_gpu=settings.docling_use_gpu)
        result = loader.load_bytes(data, filename)
        await cognee.add(result.markdown, dataset_name=dataset_name)
        logger.info("Added bytes '%s' (%d chars)", filename, len(result.markdown))
        return {"status": "added", "file": filename, "chars": len(result.markdown)}

    async def cognify(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Run Cognee ECL pipeline: extract entities, build graph, embed."""
        kwargs: dict[str, Any] = {}
        if dataset_name:
            kwargs["datasets"] = [dataset_name]
        result = await cognee.cognify(**kwargs)
        logger.info("Cognify completed: %s", result)
        return result

    async def search(
        self,
        query: str,
        search_type: str = "CHUNKS",
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search Cognee knowledge graph."""
        st = SearchType(search_type) if isinstance(search_type, str) else search_type
        raw_results = await cognee.search(query, query_type=st, top_k=limit)
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

    async def reset(self) -> None:
        """Reset all Cognee data."""
        await cognee.prune.prune_data()
        logger.info("Data reset completed")

    async def reset_system(self) -> None:
        """Full system reset including metadata."""
        await cognee.prune.prune_system(metadata=True)
        logger.info("System reset completed")
