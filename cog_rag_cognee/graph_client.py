"""Neo4j driver wrapper for direct graph queries."""
from __future__ import annotations

import logging

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphClient:
    """Lightweight wrapper for direct Neo4j access via Cypher."""

    def __init__(self, uri: str, username: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            logger.warning("Neo4j health check failed", exc_info=True)
            return False

    def get_entities(
        self, limit: int = 200, entity_types: list[str] | None = None
    ) -> list[dict]:
        """Fetch nodes that have a name property.

        Returns list of ``{id, label, type}`` dicts.
        """
        if entity_types:
            cypher = (
                "MATCH (n) WHERE n.name IS NOT NULL "
                "AND labels(n)[0] IN $types "
                "RETURN id(n) AS id, n.name AS label, labels(n)[0] AS type "
                "LIMIT $limit"
            )
            params = {"types": entity_types, "limit": limit}
        else:
            cypher = (
                "MATCH (n) WHERE n.name IS NOT NULL "
                "RETURN id(n) AS id, n.name AS label, labels(n)[0] AS type "
                "LIMIT $limit"
            )
            params = {"limit": limit}

        with self._driver.session() as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def get_relationships(
        self, limit: int = 500, entity_types: list[str] | None = None
    ) -> list[dict]:
        """Fetch relationships between named nodes.

        Returns list of ``{source, target, type}`` dicts.
        """
        if entity_types:
            cypher = (
                "MATCH (s)-[r]->(t) "
                "WHERE s.name IS NOT NULL AND t.name IS NOT NULL "
                "AND (labels(s)[0] IN $types OR labels(t)[0] IN $types) "
                "RETURN s.name AS source, t.name AS target, type(r) AS type "
                "LIMIT $limit"
            )
            params = {"types": entity_types, "limit": limit}
        else:
            cypher = (
                "MATCH (s)-[r]->(t) "
                "WHERE s.name IS NOT NULL AND t.name IS NOT NULL "
                "RETURN s.name AS source, t.name AS target, type(r) AS type "
                "LIMIT $limit"
            )
            params = {"limit": limit}

        with self._driver.session() as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def get_stats(self) -> dict:
        """Return node/edge counts and entity type breakdown.

        Returns ``{nodes, edges, entity_types: {label: count}}``.
        """
        entity_types: dict[str, int] = {}
        total_edges = 0

        with self._driver.session() as session:
            # Node counts by label
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt"
            )
            for record in result:
                entity_types[record["label"]] = record["cnt"]

            # Total edges
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            row = result.single()
            total_edges = row["cnt"] if row else 0

        return {
            "nodes": sum(entity_types.values()),
            "edges": total_edges,
            "entity_types": entity_types,
        }
