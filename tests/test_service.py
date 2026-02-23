"""Tests for PipelineService (Cognee wrapper)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_cognee():
    """Mock cognee module."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.add = AsyncMock()
        mock.cognify = AsyncMock(return_value={"main": MagicMock(
            status="completed", chunks_processed=5, entities_extracted=3
        )})
        mock.search = AsyncMock(return_value=[
            MagicMock(content="Test answer", relevance_score=0.9)
        ])
        mock.prune = MagicMock()
        mock.prune.prune_data = AsyncMock()
        mock.prune.prune_system = AsyncMock()
        yield mock


@pytest.mark.asyncio
async def test_add_text(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.add_text("Hello world")

    mock_cognee.add.assert_called_once_with("Hello world", dataset_name="main")
    assert result["status"] == "added"
    assert result["chars"] == 11


@pytest.mark.asyncio
async def test_cognify(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.cognify()

    mock_cognee.cognify.assert_called_once()
    assert result["main"].status == "completed"


@pytest.mark.asyncio
async def test_search(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    results = await svc.search("What is Cognee?")

    mock_cognee.search.assert_called_once()
    assert len(results) == 1
    assert results[0].content == "Test answer"
    assert results[0].score == 0.9


@pytest.mark.asyncio
async def test_query_full_pipeline(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    qa = await svc.query("What is Cognee?", search_type="GRAPH_COMPLETION")

    assert qa.answer == "Test answer"
    assert qa.confidence == 0.9
    assert qa.mode == "GRAPH_COMPLETION"


@pytest.mark.asyncio
async def test_add_file_uses_docling(mock_cognee, tmp_path):
    """add_file delegates to DoclingLoader then calls cognee.add."""
    f = tmp_path / "doc.txt"
    f.write_text("File content via loader")

    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.add_file(str(f))

    mock_cognee.add.assert_called_once_with("File content via loader", dataset_name="main")
    assert result["status"] == "added"
    assert result["chars"] == len("File content via loader")


@pytest.mark.asyncio
async def test_docling_loader_reused_across_calls(mock_cognee):
    """DoclingLoader is created once and reused, not per-call."""
    with patch("cog_rag_cognee.service._get_docling_loader") as mock_get_loader:
        mock_loader = MagicMock()
        mock_loader.load_bytes.return_value = MagicMock(markdown="content")
        mock_get_loader.return_value = mock_loader

        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        await svc.add_bytes(b"data1", "a.txt")
        await svc.add_bytes(b"data2", "b.txt")

        # _get_docling_loader should be called twice but return cached instance
        assert mock_get_loader.call_count == 2


@pytest.mark.asyncio
async def test_add_bytes(mock_cognee):
    """add_bytes converts bytes via DoclingLoader then calls cognee.add."""
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.add_bytes(b"Bytes content", "note.txt")

    mock_cognee.add.assert_called_once_with("Bytes content", dataset_name="main")
    assert result["status"] == "added"
    assert result["file"] == "note.txt"


@pytest.mark.asyncio
async def test_search_v052_list_format():
    """v0.5.2 returns list[list[dict]] from cognee.search."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.search = AsyncMock(return_value=[
            [{"id": "abc", "text": "Cognee is a knowledge engine", "type": "IndexSchema"}],
            [{"id": "def", "text": "It uses graphs", "type": "IndexSchema"}],
        ])
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        results = await svc.search("What is Cognee?")

    assert len(results) == 2
    assert results[0].content == "Cognee is a knowledge engine"
    assert results[1].content == "It uses graphs"
    assert results[0].score == 0.5  # default when no relevance_score


@pytest.mark.asyncio
async def test_reset(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    await svc.reset()

    mock_cognee.prune.prune_data.assert_called_once()


# --- Exception handling tests ---


@pytest.mark.asyncio
async def test_add_text_raises_ingestion_error():
    """add_text wraps cognee errors in IngestionError."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.add = AsyncMock(side_effect=RuntimeError("connection lost"))
        from cog_rag_cognee.exceptions import IngestionError
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        with pytest.raises(IngestionError, match="Failed to add text"):
            await svc.add_text("hello")


@pytest.mark.asyncio
async def test_cognify_raises_ingestion_error():
    """cognify wraps cognee errors in IngestionError."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.cognify = AsyncMock(side_effect=RuntimeError("pipeline failed"))
        from cog_rag_cognee.exceptions import IngestionError
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        with pytest.raises(IngestionError, match="Cognify failed"):
            await svc.cognify()


@pytest.mark.asyncio
async def test_search_raises_search_error():
    """search wraps cognee errors in SearchError."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.search = AsyncMock(side_effect=RuntimeError("index missing"))
        from cog_rag_cognee.exceptions import SearchError
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        with pytest.raises(SearchError, match="Search failed"):
            await svc.search("test")


@pytest.mark.asyncio
async def test_reset_raises_ingestion_error():
    """reset wraps prune errors in IngestionError."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.prune = MagicMock()
        mock.prune.prune_data = AsyncMock(side_effect=RuntimeError("storage locked"))
        from cog_rag_cognee.exceptions import IngestionError
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        with pytest.raises(IngestionError, match="Data reset failed"):
            await svc.reset()
