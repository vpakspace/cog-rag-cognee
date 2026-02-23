"""Tests for structured logging configuration."""
from __future__ import annotations

import json
import logging

from cog_rag_cognee.logging_config import setup_logging


def test_setup_logging_json_format(capsys):
    """JSON mode emits valid JSON lines."""
    setup_logging(json_logs=True, log_level="INFO")
    logger = logging.getLogger("test_json")
    logger.info("hello %s", "world")

    captured = capsys.readouterr()
    line = captured.err.strip().split("\n")[-1]
    data = json.loads(line)
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert "timestamp" in data


def test_setup_logging_text_format(capsys):
    """Text mode emits human-readable lines."""
    setup_logging(json_logs=False, log_level="DEBUG")
    logger = logging.getLogger("test_text")
    logger.debug("debug msg")

    captured = capsys.readouterr()
    assert "debug msg" in captured.err


def test_setup_logging_respects_level(capsys):
    """Messages below threshold are not emitted."""
    setup_logging(json_logs=False, log_level="WARNING")
    logger = logging.getLogger("test_level")
    logger.info("should not appear")
    logger.warning("should appear")

    captured = capsys.readouterr()
    assert "should not appear" not in captured.err
    assert "should appear" in captured.err


def test_json_formatter_includes_request_id(capsys):
    """JSON formatter includes request_id from context var."""
    from cog_rag_cognee.request_context import request_id_var

    token = request_id_var.set("test-req-123")
    try:
        setup_logging(json_logs=True, log_level="INFO")
        logger = logging.getLogger("test_req_id")
        logger.info("trace me")

        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["request_id"] == "test-req-123"
    finally:
        request_id_var.reset(token)
