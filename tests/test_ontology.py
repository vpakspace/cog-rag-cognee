"""Tests for ontology loader."""
import os

import pytest


def test_load_owl_file():
    """Should parse OWL file and extract classes and properties."""
    from cog_rag_cognee.ontology import load_ontology

    owl_path = os.path.join(os.path.dirname(__file__), "..", "ontologies", "example.owl")
    onto = load_ontology(owl_path)
    assert "classes" in onto
    assert "properties" in onto
    assert len(onto["classes"]) == 4
    assert "Person" in onto["classes"]
    assert "Organization" in onto["classes"]
    assert len(onto["properties"]) == 3


def test_load_nonexistent_file():
    """Should raise FileNotFoundError for missing file."""
    from cog_rag_cognee.ontology import load_ontology

    with pytest.raises(FileNotFoundError):
        load_ontology("/nonexistent/file.owl")


def test_ontology_to_schema_hints():
    """Should convert ontology to Cognee-compatible schema hints."""
    from cog_rag_cognee.ontology import ontology_to_schema_hints

    onto = {
        "classes": ["Person", "Organization", "Location"],
        "properties": [
            {"name": "worksFor", "domain": "Person", "range": "Organization"},
            {"name": "locatedIn", "domain": "Organization", "range": "Location"},
        ],
    }
    hints = ontology_to_schema_hints(onto)
    assert "Person" in hints["entity_types"]
    assert any(r["name"] == "worksFor" for r in hints["relationship_types"])
