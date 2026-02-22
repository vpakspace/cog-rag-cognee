"""Tests for domain models."""

from cog_rag_cognee.models import GraphStats, IngestResult, QAResult, SearchResult


def test_search_result_creation():
    """SearchResult holds content, score, source, entities, relationships."""
    r = SearchResult(
        content="Hello world",
        score=0.85,
        source="doc1.pdf",
        entities=["Alice", "Bob"],
        relationships=["knows"],
    )
    assert r.content == "Hello world"
    assert r.score == 0.85
    assert r.source == "doc1.pdf"
    assert len(r.entities) == 2
    assert len(r.relationships) == 1


def test_qa_result_creation():
    """QAResult contains answer, confidence, sources list, and mode."""
    src = SearchResult(content="chunk", score=0.9)
    qa = QAResult(
        answer="42",
        confidence=0.95,
        sources=[src],
        mode="cognitive",
    )
    assert qa.answer == "42"
    assert qa.confidence == 0.95
    assert len(qa.sources) == 1
    assert qa.mode == "cognitive"


def test_ingest_result():
    """IngestResult tracks filename, counts, status, and optional error."""
    ok = IngestResult(filename="test.pdf", chunks_added=10, entities_extracted=5)
    assert ok.status == "success"
    assert ok.error is None

    fail = IngestResult(filename="bad.pdf", status="error", error="parse failed")
    assert fail.status == "error"
    assert fail.error == "parse failed"


def test_graph_stats():
    """GraphStats holds node/edge counts and entity type breakdown."""
    gs = GraphStats(
        nodes=100,
        edges=250,
        entity_types={"Person": 40, "Organization": 30, "Location": 30},
    )
    assert gs.nodes == 100
    assert gs.edges == 250
    assert len(gs.entity_types) == 3
    assert gs.entity_types["Person"] == 40
