"""ExtractionAgent — parse PDF text and score extraction quality.

Strategy:
1. Check the cache first; if hit, load and skip extraction.
2. Try pdfplumber (fast, works well for text-based PDFs).
3. Compute quality score = pages with >50 chars / total pages.
4. If quality < 0.8, retry with pypdfium2 as fallback (when available).
5. Save the best result to cache.

pypdfium2 is an optional dependency.  Install it with:
    uv add pypdfium2
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
from rich.console import Console
from rich.table import Table

try:
    import pypdfium2 as pdfium  # type: ignore[import-untyped]
    _PYPDFIUM2_AVAILABLE = True
except ImportError:
    pdfium = None  # type: ignore[assignment]
    _PYPDFIUM2_AVAILABLE = False

from .base import BaseAgent
from ..cache import load_cache, save_cache, _pdf_hash
from ..pipeline import PageContent, PDFDocument, PipelineContext

logger = logging.getLogger(__name__)
console = Console()

QUALITY_THRESHOLD = 0.8
MIN_CHARS_PER_PAGE = 50


class ExtractionAgent(BaseAgent):
    name = "Extraction Agent"

    def _run(self, context: PipelineContext) -> PipelineContext:
        pdf_path = context.pdf_path

        # --- Cache check -------------------------------------------------------
        cached = load_cache(pdf_path)
        if cached is not None:
            doc, pdf_hash, quality = cached
            context.pdf_document = doc
            context.pdf_hash = pdf_hash
            context.extraction_quality = quality
            context.cache_hit = True
            logger.info(
                "Cache hit: %s — %d pages, quality=%.0f%%",
                pdf_path.name, doc.total_pages, quality * 100,
            )
            console.print(
                f"[green]Cache hit[/green] — loaded {doc.total_pages} pages "
                f"(quality={quality:.0%}) from cache."
            )
            return context

        # --- Fresh extraction --------------------------------------------------
        pdf_hash = _pdf_hash(pdf_path)
        context.pdf_hash = pdf_hash
        context.cache_hit = False

        logger.info("Extracting PDF: %s (hash=%s)", pdf_path.name, pdf_hash)
        console.print(f"Extracting: [cyan]{pdf_path.name}[/cyan]")

        doc, quality = self._extract_pdfplumber(pdf_path)
        method = "pdfplumber"
        logger.info("pdfplumber: %d pages, quality=%.0f%%", doc.total_pages, quality * 100)

        if quality < QUALITY_THRESHOLD:
            logger.warning(
                "pdfplumber quality %.0f%% below threshold %.0f%% — retrying with pypdfium2",
                quality * 100, QUALITY_THRESHOLD * 100,
            )
            console.print(
                f"[yellow]pdfplumber quality {quality:.0%} < {QUALITY_THRESHOLD:.0%} "
                f"— retrying with pypdfium2[/yellow]"
            )
            doc2, quality2 = self._extract_pypdfium2(pdf_path)
            if quality2 > quality:
                doc, quality = doc2, quality2
                method = "pypdfium2"
                logger.info("pypdfium2 improved quality to %.0f%%", quality * 100)

        logger.info("Extraction complete: method=%s, pages=%d, quality=%.0f%%",
                    method, doc.total_pages, quality * 100)
        self._print_quality_report(doc, quality, method)

        save_cache(pdf_path, pdf_hash, doc, quality)

        context.pdf_document = doc
        context.extraction_quality = quality
        return context

    # ------------------------------------------------------------------
    # Extraction backends
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdfplumber(pdf_path: Path) -> tuple[PDFDocument, float]:
        pages: list[PageContent] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                pages.append(PageContent(page_number=i, text=text))

        doc = PDFDocument(path=pdf_path, pages=pages)
        return doc, _quality_score(pages)

    @staticmethod
    def _extract_pypdfium2(pdf_path: Path) -> tuple[PDFDocument, float]:
        if not _PYPDFIUM2_AVAILABLE:
            logger.warning(
                "pypdfium2 is not installed — skipping fallback extraction. "
                "To enable it: uv add pypdfium2"
            )
            return PDFDocument(path=pdf_path, pages=[]), 0.0
        try:
            pages: list[PageContent] = []
            pdf = pdfium.PdfDocument(str(pdf_path))
            for i, page in enumerate(pdf, start=1):
                textpage = page.get_textpage()
                text = (textpage.get_text_range() or "").strip()
                pages.append(PageContent(page_number=i, text=text))

            doc = PDFDocument(path=pdf_path, pages=pages)
            return doc, _quality_score(pages)

        except Exception as exc:
            logger.warning("pypdfium2 extraction failed: %s", exc)
            return PDFDocument(path=pdf_path, pages=[]), 0.0

    # ------------------------------------------------------------------
    # Quality reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_quality_report(
        doc: PDFDocument, quality: float, method: str
    ) -> None:
        empty = sum(1 for p in doc.pages if len(p.text) < MIN_CHARS_PER_PAGE)
        avg_chars = (
            sum(len(p.text) for p in doc.pages) / doc.total_pages
            if doc.total_pages
            else 0
        )
        color = "green" if quality >= QUALITY_THRESHOLD else "yellow"

        table = Table(title="Extraction Quality Report", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Method", method)
        table.add_row("Total pages", str(doc.total_pages))
        table.add_row("Empty pages", str(empty))
        table.add_row("Avg chars/page", f"{avg_chars:.0f}")
        table.add_row("Quality score", f"[{color}]{quality:.0%}[/{color}]")
        console.print(table)


def _quality_score(pages: list[PageContent]) -> float:
    if not pages:
        return 0.0
    filled = sum(1 for p in pages if len(p.text) >= MIN_CHARS_PER_PAGE)
    return filled / len(pages)
