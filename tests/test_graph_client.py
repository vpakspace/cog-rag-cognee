"""Tests for GraphClient — async Neo4j driver wrapper."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class AsyncResultMock:
    """Mock for neo4j AsyncResult supporting async iteration and single()."""

    def __init__(self, records: list):
        self._records = list(records)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._records:
            raise StopAsyncIteration
        return self._records.pop(0)

    async def single(self):
        return self._records[0] if self._records else None


@pytest.fixture
def mock_driver():
    """Create a mock async Neo4j driver."""
    driver = MagicMock()
    session = AsyncMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    driver.close = AsyncMock()
    return driver, session


@pytest.fixture
def graph_client(mock_driver):
    """Create GraphClient with mocked async driver."""
    driver, _ = mock_driver
    with patch("cog_rag_cognee.graph_client.AsyncGraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = driver
        from cog_rag_cognee.graph_client import GraphClient

        client = GraphClient("neo4j://localhost:7687", "neo4j", "password")
    return client


class TestGetEntities:
    @pytest.mark.asyncio
    async def test_returns_entities(self, graph_client, mock_driver):
        _, session = mock_driver
        sample = [
            {"id": 1, "label": "Alice", "type": "Person"},
            {"id": 2, "label": "Acme Corp", "type": "Organization"},
        ]
        session.run.return_value = AsyncResultMock(list(sample))

        result = await graph_client.get_entities(limit=100)

        assert len(result) == 2
        assert result[0]["label"] == "Alice"
        assert result[1]["type"] == "Organization"
        session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock([])

        result = await graph_client.get_entities()

        assert result == []

    @pytest.mark.asyncio
    async def test_entity_type_filter(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock(
            [{"id": 1, "label": "Alice", "type": "Person"}]
        )

        await graph_client.get_entities(limit=50, entity_types=["Person"])

        call_args = session.run.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]
        assert "IN $types" in cypher
        assert params["types"] == ["Person"]
        assert params["limit"] == 50

    @pytest.mark.asyncio
    async def test_no_filter_query(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock([])

        await graph_client.get_entities(limit=200)

        call_args = session.run.call_args
        cypher = call_args[0][0]
        assert "IN $types" not in cypher


class TestGetRelationships:
    @pytest.mark.asyncio
    async def test_returns_relationships(self, graph_client, mock_driver):
        _, session = mock_driver
        sample = [
            {"source": "Alice", "target": "Bob", "type": "KNOWS"},
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT"},
        ]
        session.run.return_value = AsyncResultMock(list(sample))

        result = await graph_client.get_relationships(limit=100)

        assert len(result) == 2
        assert result[0]["source"] == "Alice"
        assert result[1]["type"] == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock([])

        result = await graph_client.get_relationships()

        assert result == []

    @pytest.mark.asyncio
    async def test_entity_type_filter(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock([])

        await graph_client.get_relationships(limit=100, entity_types=["Person"])

        call_args = session.run.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]
        assert "IN $types" in cypher
        assert params["types"] == ["Person"]


class TestGetStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self, graph_client, mock_driver):
        _, session = mock_driver

        combined_record = {
            "types": [
                {"label": "Person", "cnt": 10},
                {"label": "Organization", "cnt": 5},
            ],
            "edges": 20,
        }
        session.run.return_value = AsyncResultMock([combined_record])

        stats = await graph_client.get_stats()

        assert stats["nodes"] == 15
        assert stats["edges"] == 20
        assert stats["entity_types"]["Person"] == 10
        assert stats["entity_types"]["Organization"] == 5

    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver

        combined_record = {"types": [], "edges": 0}
        session.run.return_value = AsyncResultMock([combined_record])

        stats = await graph_client.get_stats()

        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["entity_types"] == {}


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_success(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = AsyncResultMock([])

        assert await graph_client.health_check() is True

    @pytest.mark.asyncio
    async def test_failure(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("Connection refused")

        assert await graph_client.health_check() is False


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, graph_client, mock_driver):
        driver, _ = mock_driver
        await graph_client.close()
        driver.close.assert_called_once()
