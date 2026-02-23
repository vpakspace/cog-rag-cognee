"""Tests for DoclingLoader — document loading with optional Docling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cog_rag_cognee.docling_loader import DoclingLoader, load_file

# ── Plain text (no Docling required) ─────────────────────────────


def test_load_txt_returns_markdown(tmp_path: Path):
    """Loading .txt returns content without Docling."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello world", encoding="utf-8")

    loader = DoclingLoader()
    result = loader.load(f)

    assert result.markdown == "Hello world"
    assert result.metadata["format"] == ".txt"
    assert result.metadata["pages"] == 1


def test_load_md_returns_markdown(tmp_path: Path):
    """Loading .md returns content without Docling."""
    f = tmp_path / "readme.md"
    f.write_text("# Title\nBody text", encoding="utf-8")

    loader = DoclingLoader()
    result = loader.load(f)

    assert result.markdown == "# Title\nBody text"
    assert result.metadata["format"] == ".md"


def test_load_txt_bytes():
    """load_bytes for plain text decodes in memory."""
    loader = DoclingLoader()
    result = loader.load_bytes(b"Hello bytes", "note.txt")

    assert result.markdown == "Hello bytes"
    assert result.metadata["format"] == ".txt"


def test_load_md_bytes():
    """load_bytes for .md decodes in memory."""
    loader = DoclingLoader()
    result = loader.load_bytes(b"# Header", "doc.md")

    assert result.markdown == "# Header"


# ── Validation ────────────────────────────────────────────────────


def test_load_unsupported_extension_raises(tmp_path: Path):
    """Unsupported extension raises ValueError."""
    f = tmp_path / "data.xyz"
    f.write_text("data")

    loader = DoclingLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load(f)


def test_load_bytes_unsupported_raises():
    """load_bytes with unsupported extension raises ValueError."""
    loader = DoclingLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load_bytes(b"data", "file.xyz")


def test_load_missing_file_raises():
    """Non-existent file raises FileNotFoundError."""
    loader = DoclingLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/file.pdf")


# ── Docling integration (mocked) ────────────────────────────────


def _mock_docling_modules():
    """Create mock modules that simulate docling imports."""
    mock_converter_cls = MagicMock()
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = "# Converted PDF content"
    mock_doc.num_pages = 3

    mock_converter_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.document = mock_doc
    mock_converter_instance.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter_instance

    return {
        "docling": MagicMock(),
        "docling.datamodel": MagicMock(),
        "docling.datamodel.base_models": MagicMock(InputFormat=MagicMock(PDF="pdf")),
        "docling.datamodel.pipeline_options": MagicMock(
            PdfPipelineOptions=MagicMock(return_value=MagicMock())
        ),
        "docling.document_converter": MagicMock(
            DocumentConverter=mock_converter_cls,
            PdfFormatOption=MagicMock(),
        ),
    }


def test_load_pdf_calls_docling(tmp_path: Path):
    """Loading PDF delegates to Docling converter."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    mocks = _mock_docling_modules()
    with patch.dict("sys.modules", mocks):
        # Force fresh converter
        loader = DoclingLoader()
        result = loader.load(f)

    assert result.markdown == "# Converted PDF content"
    assert result.metadata["format"] == ".pdf"


def test_load_docx_calls_docling(tmp_path: Path):
    """Loading DOCX delegates to Docling converter."""
    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK fake docx")

    mocks = _mock_docling_modules()
    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader()
        result = loader.load(f)

    assert result.markdown == "# Converted PDF content"
    assert result.metadata["format"] == ".docx"


def test_load_bytes_pdf_uses_tempfile(tmp_path: Path):
    """load_bytes for PDF creates a temp file and cleans up."""
    mocks = _mock_docling_modules()
    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader()
        result = loader.load_bytes(b"%PDF-1.4 fake", "report.pdf")

    assert result.markdown == "# Converted PDF content"


# ── Converter lifecycle ──────────────────────────────────────────


def test_gpu_disabled_by_default():
    """GPU is off by default."""
    loader = DoclingLoader()
    assert loader._use_gpu is False


def test_gpu_enabled_when_requested():
    """GPU flag is set when requested."""
    loader = DoclingLoader(use_gpu=True)
    assert loader._use_gpu is True


def test_converter_not_initialized_on_construction():
    """Converter is None until first binary load."""
    loader = DoclingLoader()
    assert loader._converter is None


def test_converter_initialized_on_first_load(tmp_path: Path):
    """After loading a binary file, converter is initialized."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")

    mocks = _mock_docling_modules()
    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader()
        loader.load(f)

    assert loader._converter is not None


def test_plain_text_does_not_init_converter(tmp_path: Path):
    """Loading .txt does not trigger converter initialization."""
    f = tmp_path / "note.txt"
    f.write_text("text content")

    loader = DoclingLoader()
    loader.load(f)

    assert loader._converter is None


# ── Import error fallback ────────────────────────────────────────


def test_import_error_fallback(tmp_path: Path):
    """Missing docling raises ImportError with helpful message."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")

    with patch.dict("sys.modules", {"docling": None}):
        loader = DoclingLoader()
        with pytest.raises(ImportError, match="pip install docling"):
            loader.load(f)


# ── Convenience function ─────────────────────────────────────────


def test_load_file_convenience(tmp_path: Path):
    """load_file() returns plain string."""
    f = tmp_path / "data.txt"
    f.write_text("convenience content")

    text = load_file(str(f))
    assert text == "convenience content"
    assert isinstance(text, str)


# ── GPU acceleration path (lines 76-87) ───────────────────────


def _mock_docling_modules_with_gpu():
    """Create mock modules that simulate docling + GPU acceleration imports."""
    mock_converter_cls = MagicMock()
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = "# GPU content"
    mock_doc.num_pages = 5

    mock_converter_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.document = mock_doc
    mock_converter_instance.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter_instance

    mock_accel_device = MagicMock()
    mock_accel_device.AUTO = "auto"
    mock_accel_options = MagicMock()

    return {
        "docling": MagicMock(),
        "docling.datamodel": MagicMock(),
        "docling.datamodel.base_models": MagicMock(InputFormat=MagicMock(PDF="pdf")),
        "docling.datamodel.pipeline_options": MagicMock(
            PdfPipelineOptions=MagicMock(return_value=MagicMock())
        ),
        "docling.document_converter": MagicMock(
            DocumentConverter=mock_converter_cls,
            PdfFormatOption=MagicMock(),
        ),
        "docling.datamodel.accelerator_options": MagicMock(
            AcceleratorDevice=mock_accel_device,
            AcceleratorOptions=mock_accel_options,
        ),
    }


def test_gpu_acceleration_enabled(tmp_path: Path):
    """GPU path imports AcceleratorDevice and sets options (lines 76-85)."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    mocks = _mock_docling_modules_with_gpu()
    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader(use_gpu=True)
        result = loader.load(f)

    assert result.markdown == "# GPU content"
    # Verify AcceleratorOptions was called
    accel_mod = mocks["docling.datamodel.accelerator_options"]
    accel_mod.AcceleratorOptions.assert_called_once()


def test_gpu_import_fallback(tmp_path: Path):
    """GPU falls back to CPU when accelerator imports fail (lines 86-89)."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    mocks = _mock_docling_modules()
    # Simulate accelerator_options import failure
    mocks["docling.datamodel.accelerator_options"] = None
    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader(use_gpu=True)
        result = loader.load(f)

    # Should succeed with CPU fallback
    assert result.markdown == "# Converted PDF content"


# ── Callable pages() branch (line 143) ──────────────────────


def test_callable_num_pages(tmp_path: Path):
    """When doc.num_pages is callable, it gets called (line 143)."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    mocks = _mock_docling_modules()
    # Make num_pages a callable that returns 7
    doc_mock = mocks["docling.document_converter"].DocumentConverter.return_value \
        .convert.return_value.document
    doc_mock.num_pages = MagicMock(return_value=7)
    # Make callable() return True for num_pages
    doc_mock.num_pages.__call__ = MagicMock(return_value=7)

    with patch.dict("sys.modules", mocks):
        loader = DoclingLoader()
        result = loader.load(f)

    assert result.metadata["pages"] == 7
