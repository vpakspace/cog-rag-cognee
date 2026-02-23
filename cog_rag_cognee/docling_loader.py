"""Document loader with optional IBM Docling support.

Plain text files (.txt, .md) are handled directly without Docling.
Binary formats (.pdf, .docx, .pptx, .xlsx, .html) require Docling to be
installed: ``pip install docling``.

Docling is imported lazily — the heavy models (~1-2 GB) are only loaded
on the first call that actually needs them.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"}
_PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}


@dataclass
class LoadResult:
    """Result of document loading.

    Attributes:
        markdown: Document content as markdown text.
        metadata: File-level metadata (format, pages).
    """

    markdown: str
    metadata: dict = field(default_factory=dict)


class DoclingLoader:
    """Document loader with lazy Docling initialization and optional GPU.

    Args:
        use_gpu: When ``True``, PDF processing uses ``AcceleratorDevice.AUTO``.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self._converter: DocumentConverter | None = None
        self._use_gpu = use_gpu

    def _get_converter(self) -> DocumentConverter:
        """Lazy-initialize the Docling DocumentConverter.

        Raises:
            ImportError: When Docling is not installed.
        """
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError:
                raise ImportError(
                    "Docling is required for binary document formats. "
                    "Install it with: pip install docling"
                ) from None

            pipeline_options = PdfPipelineOptions()

            if self._use_gpu:
                try:
                    from docling.datamodel.accelerator_options import (
                        AcceleratorDevice,
                        AcceleratorOptions,
                    )

                    pipeline_options.accelerator_options = AcceleratorOptions(
                        device=AcceleratorDevice.AUTO
                    )
                    logger.info("GPU acceleration enabled (AcceleratorDevice.AUTO)")
                except ImportError:
                    logger.warning(
                        "GPU acceleration imports unavailable — falling back to CPU"
                    )

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        return self._converter

    def load(self, file_path: str | Path) -> LoadResult:
        """Load a document from disk.

        Args:
            file_path: Path to a supported document file.

        Returns:
            :class:`LoadResult` with markdown text and metadata.

        Raises:
            FileNotFoundError: When *file_path* does not exist.
            ValueError: When the file extension is unsupported.
            ImportError: When Docling is needed but not installed.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Plain text — no Docling needed
        if ext in _PLAIN_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            logger.debug("Loaded plain text from %s (%d chars)", path.name, len(text))
            return LoadResult(
                markdown=text,
                metadata={"format": ext, "pages": 1},
            )

        # Binary formats — delegate to Docling
        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document
        markdown = doc.export_to_markdown()

        pages = getattr(doc, "num_pages", None)
        if callable(pages):
            pages = pages()

        logger.info("Loaded %d chars from %s", len(markdown), path.name)
        return LoadResult(
            markdown=markdown,
            metadata={"format": ext, "pages": pages},
        )

    def load_bytes(self, data: bytes, filename: str) -> LoadResult:
        """Load a document from raw bytes (for upload handlers).

        Args:
            data:     Raw file bytes.
            filename: Original filename (used for extension detection).

        Returns:
            :class:`LoadResult` (same as :meth:`load`).

        Raises:
            ValueError: When the extension is unsupported.
            ImportError: When Docling is needed but not installed.
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Plain text — decode in memory
        if ext in _PLAIN_TEXT_EXTENSIONS:
            text = data.decode("utf-8", errors="replace")
            logger.debug(
                "Loaded plain text bytes for %s (%d chars)", filename, len(text)
            )
            return LoadResult(
                markdown=text,
                metadata={"format": ext, "pages": 1},
            )

        # Binary formats — write to temp file, then load
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        path = Path(tmp_path)
        try:
            os.close(fd)
            path.write_bytes(data)
            return self.load(path)
        finally:
            path.unlink(missing_ok=True)


def load_file(file_path: str, use_gpu: bool = False) -> str:
    """Convenience: load a document and return markdown text.

    Args:
        file_path: Path to a supported document file.
        use_gpu:   Enable GPU acceleration for PDF processing.

    Returns:
        Markdown-formatted text content.
    """
    loader = DoclingLoader(use_gpu=use_gpu)
    return loader.load(file_path).markdown
