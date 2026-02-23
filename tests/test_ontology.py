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


def test_local_name_with_slash_uri():
    """_local_name extracts name from URI using / separator (no #)."""
    from cog_rag_cognee.ontology import _local_name

    assert _local_name("http://example.org/ontology/Person") == "Person"
    assert _local_name("http://example.org/a/b/Thing") == "Thing"


def test_local_name_with_hash_uri():
    """_local_name extracts name from URI using # separator."""
    from cog_rag_cognee.ontology import _local_name

    assert _local_name("http://example.org/ontology#Person") == "Person"


def test_property_without_about(tmp_path):
    """Properties without rdf:about are skipped."""
    from cog_rag_cognee.ontology import load_ontology

    owl_content = """\
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Class rdf:about="http://example.org#Cat"/>
  <owl:ObjectProperty/>
  <owl:ObjectProperty rdf:about="http://example.org#chases">
    <rdfs:domain rdf:resource="http://example.org#Cat"/>
  </owl:ObjectProperty>
</rdf:RDF>
"""
    owl_file = tmp_path / "test.owl"
    owl_file.write_text(owl_content)
    onto = load_ontology(str(owl_file))
    # The property without rdf:about is skipped
    assert len(onto["properties"]) == 1
    assert onto["properties"][0]["name"] == "chases"


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
