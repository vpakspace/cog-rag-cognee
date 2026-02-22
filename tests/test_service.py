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
async def test_add_bytes(mock_cognee):
    """add_bytes converts bytes via DoclingLoader then calls cognee.add."""
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.add_bytes(b"Bytes content", "note.txt")

    mock_cognee.add.assert_called_once_with("Bytes content", dataset_name="main")
    assert result["status"] == "added"
    assert result["file"] == "note.txt"


@pytest.mark.asyncio
async def test_reset(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    await svc.reset()

    mock_cognee.prune.prune_data.assert_called_once()
