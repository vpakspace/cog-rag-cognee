"""Tests for GraphClient — Neo4j driver wrapper."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def graph_client(mock_driver):
    """Create GraphClient with mocked driver."""
    driver, _ = mock_driver
    with patch("cog_rag_cognee.graph_client.GraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = driver
        from cog_rag_cognee.graph_client import GraphClient

        client = GraphClient("neo4j://localhost:7687", "neo4j", "password")
    return client


class TestGetEntities:
    def test_returns_entities(self, graph_client, mock_driver):
        _, session = mock_driver
        sample = [
            {"id": 1, "label": "Alice", "type": "Person"},
            {"id": 2, "label": "Acme Corp", "type": "Organization"},
        ]
        session.run.return_value = sample

        result = graph_client.get_entities(limit=100)

        assert len(result) == 2
        assert result[0]["label"] == "Alice"
        assert result[1]["type"] == "Organization"
        session.run.assert_called_once()

    def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = []

        result = graph_client.get_entities()

        assert result == []

    def test_entity_type_filter(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = [{"id": 1, "label": "Alice", "type": "Person"}]

        graph_client.get_entities(limit=50, entity_types=["Person"])

        call_args = session.run.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]
        assert "IN $types" in cypher
        assert params["types"] == ["Person"]
        assert params["limit"] == 50

    def test_no_filter_query(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = []

        graph_client.get_entities(limit=200)

        call_args = session.run.call_args
        cypher = call_args[0][0]
        assert "IN $types" not in cypher


class TestGetRelationships:
    def test_returns_relationships(self, graph_client, mock_driver):
        _, session = mock_driver
        sample = [
            {"source": "Alice", "target": "Bob", "type": "KNOWS"},
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT"},
        ]
        session.run.return_value = sample

        result = graph_client.get_relationships(limit=100)

        assert len(result) == 2
        assert result[0]["source"] == "Alice"
        assert result[1]["type"] == "WORKS_AT"

    def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = []

        result = graph_client.get_relationships()

        assert result == []

    def test_entity_type_filter(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = []

        graph_client.get_relationships(limit=100, entity_types=["Person"])

        call_args = session.run.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]
        assert "IN $types" in cypher
        assert params["types"] == ["Person"]


class TestGetStats:
    def test_returns_stats(self, graph_client, mock_driver):
        _, session = mock_driver

        # Single combined query returns types list + edges count
        combined_result = MagicMock()
        combined_result.single.return_value = {
            "types": [
                {"label": "Person", "cnt": 10},
                {"label": "Organization", "cnt": 5},
            ],
            "edges": 20,
        }
        session.run.return_value = combined_result

        stats = graph_client.get_stats()

        assert stats["nodes"] == 15
        assert stats["edges"] == 20
        assert stats["entity_types"]["Person"] == 10
        assert stats["entity_types"]["Organization"] == 5

    def test_empty_graph(self, graph_client, mock_driver):
        _, session = mock_driver

        combined_result = MagicMock()
        combined_result.single.return_value = {
            "types": [],
            "edges": 0,
        }
        session.run.return_value = combined_result

        stats = graph_client.get_stats()

        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["entity_types"] == {}


class TestHealthCheck:
    def test_success(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.return_value = MagicMock()

        assert graph_client.health_check() is True

    def test_failure(self, graph_client, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("Connection refused")

        assert graph_client.health_check() is False
