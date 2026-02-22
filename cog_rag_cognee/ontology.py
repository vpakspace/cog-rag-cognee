"""OWL/RDF ontology loader for domain grounding."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"


def _local_name(uri: str) -> str:
    """Extract local name from URI (after # or last /)."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rsplit("/", 1)[-1]


def load_ontology(file_path: str) -> dict[str, Any]:
    """Parse OWL/RDF file and extract classes and properties."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {file_path}")

    tree = ET.parse(path)
    root = tree.getroot()

    classes = []
    for cls in root.findall(f".//{{{OWL_NS}}}Class"):
        about = cls.get(f"{{{RDF_NS}}}about", "")
        if about:
            classes.append(_local_name(about))

    properties = []
    for prop in root.findall(f".//{{{OWL_NS}}}ObjectProperty"):
        about = prop.get(f"{{{RDF_NS}}}about", "")
        if not about:
            continue
        name = _local_name(about)
        domain_el = prop.find(f"{{{RDFS_NS}}}domain")
        range_el = prop.find(f"{{{RDFS_NS}}}range")
        domain = (
            _local_name(domain_el.get(f"{{{RDF_NS}}}resource", ""))
            if domain_el is not None
            else ""
        )
        range_ = (
            _local_name(range_el.get(f"{{{RDF_NS}}}resource", ""))
            if range_el is not None
            else ""
        )
        properties.append({"name": name, "domain": domain, "range": range_})

    logger.info(
        "Loaded ontology: %d classes, %d properties from %s",
        len(classes),
        len(properties),
        file_path,
    )
    return {"classes": classes, "properties": properties}


def ontology_to_schema_hints(onto: dict[str, Any]) -> dict[str, Any]:
    """Convert parsed ontology to Cognee-compatible schema hints."""
    return {
        "entity_types": onto.get("classes", []),
        "relationship_types": onto.get("properties", []),
    }
