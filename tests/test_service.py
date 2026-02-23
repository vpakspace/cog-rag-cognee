"""Tests for PipelineService (Cognee wrapper)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cog_rag_cognee.service import retry_transient

_LOADER_PATH = "cog_rag_cognee.service._get_docling_loader"


@pytest.fixture
def svc(mock_cognee):
    """Return a PipelineService with the cognee module already mocked."""
    from cog_rag_cognee.service import PipelineService

    return PipelineService()


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
async def test_query_no_results(mock_cognee):
    """query returns fallback answer when search yields nothing."""
    mock_cognee.search = AsyncMock(return_value=[])
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    qa = await svc.query("Unknown topic")

    assert qa.answer == "No relevant information found."
    assert qa.confidence == 0.0


@pytest.mark.asyncio
async def test_list_datasets_error():
    """list_datasets returns empty list on exception."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.datasets.list_datasets = AsyncMock(side_effect=RuntimeError("db down"))
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        result = await svc.list_datasets()

    assert result == []


@pytest.mark.asyncio
async def test_list_datasets_success():
    """list_datasets returns dataset names."""
    ds1 = MagicMock()
    ds1.name = "main"
    ds2 = MagicMock()
    ds2.name = "papers"
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.datasets.list_datasets = AsyncMock(return_value=[ds1, ds2])
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        result = await svc.list_datasets()

    assert result == ["main", "papers"]


@pytest.mark.asyncio
async def test_add_file_reraises_ingestion_error():
    """add_file re-raises IngestionError without wrapping."""
    from cog_rag_cognee.exceptions import IngestionError
    from cog_rag_cognee.service import PipelineService

    inner_loader = MagicMock()
    inner_loader.load.side_effect = IngestionError("bad format")
    loader_mock = AsyncMock(return_value=inner_loader)
    with (
        patch("cog_rag_cognee.service.cognee"),
        patch(_LOADER_PATH, new=loader_mock),
    ):
        svc = PipelineService()
        with pytest.raises(IngestionError, match="bad format"):
            await svc.add_file("/tmp/bad.pdf")


@pytest.mark.asyncio
async def test_add_bytes_reraises_ingestion_error():
    """add_bytes re-raises IngestionError without wrapping."""
    from cog_rag_cognee.exceptions import IngestionError
    from cog_rag_cognee.service import PipelineService

    inner_loader = MagicMock()
    inner_loader.load_bytes.side_effect = IngestionError("corrupt")
    loader_mock = AsyncMock(return_value=inner_loader)
    with (
        patch("cog_rag_cognee.service.cognee"),
        patch(_LOADER_PATH, new=loader_mock),
    ):
        svc = PipelineService()
        with pytest.raises(IngestionError, match="corrupt"):
            await svc.add_bytes(b"data", "bad.pdf")


@pytest.mark.asyncio
async def test_cognify_reraises_ingestion_error():
    """cognify re-raises IngestionError without double-wrapping."""
    from cog_rag_cognee.exceptions import IngestionError
    from cog_rag_cognee.service import PipelineService

    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.cognify = AsyncMock(side_effect=IngestionError("pipeline error"))
        svc = PipelineService()
        with pytest.raises(IngestionError, match="pipeline error"):
            await svc.cognify()


@pytest.mark.asyncio
async def test_search_reraises_search_error():
    """search re-raises SearchError without double-wrapping."""
    from cog_rag_cognee.exceptions import SearchError
    from cog_rag_cognee.service import PipelineService

    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.search = AsyncMock(side_effect=SearchError("index broken"))
        svc = PipelineService()
        with pytest.raises(SearchError, match="index broken"):
            await svc.search("test")


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
    mock_loader = MagicMock()
    mock_loader.load_bytes.return_value = MagicMock(markdown="content")
    mock_get_loader = AsyncMock(return_value=mock_loader)
    with patch(_LOADER_PATH, new=mock_get_loader):
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


# --- _extract_result branch tests ---


def test_extract_result_plain_string():
    """Plain string input returns str with 0.5 score."""
    from cog_rag_cognee.service import PipelineService

    content, score = PipelineService._extract_result("raw text")
    assert content == "raw text"
    assert score == 0.5


def test_extract_result_empty_list():
    """Empty list returns empty string."""
    from cog_rag_cognee.service import PipelineService

    content, score = PipelineService._extract_result([])
    assert content == ""
    assert score == 0.5


def test_extract_result_list_of_strings():
    """List of strings joins them."""
    from cog_rag_cognee.service import PipelineService

    content, score = PipelineService._extract_result(["hello", "world"])
    assert content == "hello\nworld"
    assert score == 0.5


def test_extract_result_list_with_non_dict_non_str():
    """List with non-dict non-str items uses str()."""
    from cog_rag_cognee.service import PipelineService

    content, score = PipelineService._extract_result([42, None])
    assert "42" in content
    assert "None" in content


def test_extract_result_object_with_content_list():
    """Object with .content as list extracts text."""
    from cog_rag_cognee.service import PipelineService

    obj = MagicMock()
    obj.content = [{"text": "chunk1"}, "plain", 42]
    obj.relevance_score = 0.8
    content, score = PipelineService._extract_result(obj)
    assert "chunk1" in content
    assert "plain" in content
    assert "42" in content
    assert score == 0.8


def test_extract_result_object_with_content_string():
    """Object with .content as string returns it directly."""
    from cog_rag_cognee.service import PipelineService

    obj = MagicMock()
    obj.content = "direct content"
    del obj.relevance_score  # no score attr
    content, score = PipelineService._extract_result(obj)
    assert content == "direct content"
    assert score == 0.5


def test_extract_result_score_clamped():
    """Scores outside [0,1] are clamped."""
    from cog_rag_cognee.service import PipelineService

    obj = MagicMock()
    obj.content = "text"
    obj.relevance_score = 1.5
    _, score = PipelineService._extract_result(obj)
    assert score == 1.0

    obj.relevance_score = -0.3
    _, score = PipelineService._extract_result(obj)
    assert score == 0.0


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
async def test_ingestion_error_preserves_cause(svc, mock_cognee):
    """IngestionError should chain the original exception as __cause__."""
    from cog_rag_cognee.exceptions import IngestionError

    mock_cognee.add = AsyncMock(side_effect=RuntimeError("disk full"))
    with pytest.raises(IngestionError) as exc_info:
        await svc.add_text("hello")
    assert exc_info.value.__cause__ is not None
    assert "disk full" in str(exc_info.value.__cause__)


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


# --- Retry logic tests ---


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_try():
    """retry_transient returns result immediately on success."""
    func = AsyncMock(return_value="ok")
    result = await retry_transient(func, max_retries=3, base_delay=0)
    assert result == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_recovers_after_transient_failure():
    """retry_transient recovers after ConnectionError."""
    func = AsyncMock(side_effect=[ConnectionError("refused"), "ok"])
    result = await retry_transient(func, max_retries=3, base_delay=0)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """retry_transient raises after max_retries exceeded."""
    func = AsyncMock(side_effect=ConnectionError("down"))
    with pytest.raises(ConnectionError):
        await retry_transient(func, max_retries=2, base_delay=0)
    assert func.call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_retry_does_not_retry_value_error():
    """retry_transient does not retry ValueError (non-transient)."""
    func = AsyncMock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        await retry_transient(func, max_retries=3, base_delay=0)
    assert func.call_count == 1


# --- Timeout tests ---


@pytest.mark.asyncio
async def test_retry_transient_respects_timeout():
    """retry_transient raises TimeoutError when operation exceeds timeout."""
    import asyncio

    async def slow_func():
        await asyncio.sleep(10)
        return "never"

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await retry_transient(slow_func, max_retries=0, base_delay=0, timeout=0.05)


# --- Coverage gap tests: service.py lines 87-88, 106-107, 116 ---


@pytest.mark.asyncio
async def test_add_file_wraps_generic_exception():
    """add_file wraps non-IngestionError in IngestionError (lines 87-88)."""
    from cog_rag_cognee.exceptions import IngestionError
    from cog_rag_cognee.service import PipelineService

    inner_loader = MagicMock()
    inner_loader.load.return_value.markdown = "text"
    loader_mock = AsyncMock(return_value=inner_loader)
    with (
        patch("cog_rag_cognee.service.cognee") as mock_cognee,
        patch(_LOADER_PATH, new=loader_mock),
    ):
        mock_cognee.add = AsyncMock(side_effect=RuntimeError("conn lost"))
        svc = PipelineService()
        with pytest.raises(IngestionError, match="Failed to add file"):
            await svc.add_file("/tmp/test.txt")


@pytest.mark.asyncio
async def test_add_bytes_wraps_generic_exception():
    """add_bytes wraps non-IngestionError in IngestionError (lines 106-107)."""
    from cog_rag_cognee.exceptions import IngestionError
    from cog_rag_cognee.service import PipelineService

    inner_loader = MagicMock()
    inner_loader.load_bytes.return_value.markdown = "text"
    loader_mock = AsyncMock(return_value=inner_loader)
    with (
        patch("cog_rag_cognee.service.cognee") as mock_cognee,
        patch(_LOADER_PATH, new=loader_mock),
    ):
        mock_cognee.add = AsyncMock(side_effect=RuntimeError("conn lost"))
        svc = PipelineService()
        with pytest.raises(IngestionError, match="Failed to add bytes"):
            await svc.add_bytes(b"data", "note.txt")


@pytest.mark.asyncio
async def test_cognify_with_dataset_name():
    """cognify passes dataset_name as datasets kwarg (line 116)."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.cognify = AsyncMock(return_value={"papers": MagicMock(status="completed")})
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        result = await svc.cognify(dataset_name="papers")

    mock.cognify.assert_called_once()
    call_kwargs = mock.cognify.call_args[1]
    assert call_kwargs["datasets"] == ["papers"]
    assert "papers" in result


@pytest.mark.asyncio
async def test_search_invalid_search_type_raises_search_error():
    """search() wraps invalid SearchType in SearchError, not raw ValueError."""
    with patch("cog_rag_cognee.service.cognee"):
        from cog_rag_cognee.exceptions import SearchError
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        with pytest.raises(SearchError, match="Unknown search type"):
            await svc.search("test", search_type="BOGUS_TYPE")


@pytest.mark.asyncio
async def test_add_file_propagates_dataset_name():
    """add_file forwards dataset_name to cognee.add."""
    inner_loader = MagicMock()
    inner_loader.load.return_value.markdown = "content"
    loader_mock = AsyncMock(return_value=inner_loader)
    with (
        patch("cog_rag_cognee.service.cognee") as mock_cognee,
        patch(_LOADER_PATH, new=loader_mock),
    ):
        mock_cognee.add = AsyncMock()
        from cog_rag_cognee.service import PipelineService

        svc = PipelineService()
        await svc.add_file("/tmp/notes.txt", dataset_name="research")

    mock_cognee.add.assert_called_once_with("content", dataset_name="research")


def test_cleanup_docling_loader():
    """cleanup_docling_loader should release the cached loader."""
    import cog_rag_cognee.service as svc_mod
    from cog_rag_cognee.service import cleanup_docling_loader

    # Force init by setting a mock
    svc_mod._docling_loader = MagicMock()
    assert svc_mod._docling_loader is not None

    cleanup_docling_loader()
    assert svc_mod._docling_loader is None


@pytest.mark.asyncio
async def test_docling_loader_singleton_thread_safe():
    """Concurrent _get_docling_loader calls produce a single instance."""
    import asyncio

    import cog_rag_cognee.service as svc_mod
    from cog_rag_cognee.service import _get_docling_loader

    # Reset the module-level singleton so we start from a clean slate.
    original = svc_mod._docling_loader
    svc_mod._docling_loader = None
    try:
        fake_loader = MagicMock()
        with patch("cog_rag_cognee.service.DoclingLoader", return_value=fake_loader):
            # Launch several coroutines concurrently; only one should construct
            # DoclingLoader even though they all reach the None-check simultaneously.
            results = await asyncio.gather(*[_get_docling_loader() for _ in range(5)])

        # Every call returns the same singleton object.
        assert all(r is fake_loader for r in results)
    finally:
        # Restore whatever was there before (None in practice during tests).
        svc_mod._docling_loader = original
