"""Tests for benchmark runner keyword overlap judge."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_benchmark import CONCEPT_MAP, evaluate_answer


class TestEvaluateAnswer:
    """Tests for evaluate_answer function."""

    def test_pass_with_keyword_match(self):
        q = {"expected_keywords": ["knowledge", "engine", "memory"]}
        answer = "Cognee is a knowledge engine with persistent memory support."
        assert evaluate_answer(q, answer) is True

    def test_fail_empty_answer(self):
        q = {"expected_keywords": ["neo4j", "graph"]}
        assert evaluate_answer(q, "") is False

    def test_fail_short_answer(self):
        q = {"expected_keywords": ["neo4j", "graph"]}
        assert evaluate_answer(q, "yes") is False

    def test_cross_language_ru_keyword_en_answer(self):
        """RU keyword 'память' should match EN answer containing 'memory'."""
        q = {"expected_keywords": ["знаний", "память", "документы"]}
        answer = "Cognee provides knowledge management with memory and documents processing."
        assert evaluate_answer(q, answer) is True

    def test_cross_language_en_keyword_ru_answer(self):
        """EN keyword 'graph' should match RU answer containing 'граф'."""
        q = {"expected_keywords": ["graph", "storage", "entities"]}
        answer = "Cognee использует граф знаний для хранения сущностей и связей."
        assert evaluate_answer(q, answer) is True

    def test_no_keywords_long_answer(self):
        q = {"expected_keywords": []}
        answer = "This is a detailed answer that exceeds fifty characters in total length easily."
        assert evaluate_answer(q, answer) is True

    def test_no_keywords_short_answer(self):
        q = {"expected_keywords": []}
        answer = "Short but above 20 chars."
        assert evaluate_answer(q, answer) is False

    def test_partial_keyword_match_above_threshold(self):
        """1 out of 3 keywords = 0.33 overlap > 0.3 threshold → PASS."""
        q = {"expected_keywords": ["neo4j", "cypher", "driver"]}
        answer = "The system stores data in neo4j graph database for queries."
        assert evaluate_answer(q, answer) is True

    def test_zero_keyword_match(self):
        """0 out of 3 keywords → FAIL."""
        q = {"expected_keywords": ["neo4j", "cypher", "driver"]}
        answer = "The system uses a vector database for embedding storage."
        assert evaluate_answer(q, answer) is False

    def test_case_insensitive_match(self):
        q = {"expected_keywords": ["LanceDB", "Embedded"]}
        answer = "lancedb provides an embedded vector store without docker."
        assert evaluate_answer(q, answer) is True

    def test_partial_keyword_in_word(self):
        """Keyword 'сущност' should match 'сущности' or 'сущностей'."""
        q = {"expected_keywords": ["граф", "хранения", "сущност"]}
        answer = "Граф знаний хранит сущности и связи между ними."
        assert evaluate_answer(q, answer) is True


class TestConceptMap:
    """Tests for CONCEPT_MAP integrity."""

    def test_concept_map_not_empty(self):
        assert len(CONCEPT_MAP) > 0

    def test_all_values_are_english(self):
        for ru, en in CONCEPT_MAP.items():
            assert en.isascii(), f"Value '{en}' for key '{ru}' is not ASCII"
