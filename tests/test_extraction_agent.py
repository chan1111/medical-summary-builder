"""Tests for ExtractionAgent and cache utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medical_summary_builder.agents.extraction_agent import (
    ExtractionAgent,
    _quality_score,
    QUALITY_THRESHOLD,
    MIN_CHARS_PER_PAGE,
)
from medical_summary_builder.cache import load_cache, save_cache
from medical_summary_builder.pipeline import PageContent, PDFDocument, PipelineContext

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PDF = DATA_DIR / "Medical File.pdf"


# ---------------------------------------------------------------------------
# _quality_score unit tests
# ---------------------------------------------------------------------------

class TestQualityScore:
    def test_empty_pages_returns_zero(self):
        assert _quality_score([]) == 0.0

    def test_all_empty_pages(self):
        pages = [PageContent(page_number=i, text="") for i in range(1, 6)]
        assert _quality_score(pages) == 0.0

    def test_all_filled_pages(self):
        text = "x" * 100
        pages = [PageContent(page_number=i, text=text) for i in range(1, 6)]
        assert _quality_score(pages) == 1.0

    def test_half_filled_pages(self):
        pages = [
            PageContent(page_number=1, text="x" * 100),
            PageContent(page_number=2, text=""),
        ]
        assert _quality_score(pages) == 0.5

    def test_exactly_at_threshold_char_count(self):
        """Page with exactly MIN_CHARS_PER_PAGE chars should count as filled."""
        text = "x" * MIN_CHARS_PER_PAGE
        pages = [PageContent(page_number=1, text=text)]
        assert _quality_score(pages) == 1.0

    def test_one_char_below_threshold(self):
        text = "x" * (MIN_CHARS_PER_PAGE - 1)
        pages = [PageContent(page_number=1, text=text)]
        assert _quality_score(pages) == 0.0


# ---------------------------------------------------------------------------
# ExtractionAgent with real PDF (skipped if not present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not present")
def test_extraction_agent_returns_pages():
    ctx = PipelineContext(
        pdf_path=SAMPLE_PDF,
        template_path=Path("docs/Medical Summary.docx"),
        output_path=Path("output/test_summary.docx"),
    )
    agent = ExtractionAgent()
    ctx = agent.run(ctx)

    assert ctx.pdf_document is not None
    assert ctx.pdf_document.total_pages > 0
    assert 0.0 <= ctx.extraction_quality <= 1.0


# ---------------------------------------------------------------------------
# ExtractionAgent error handling
# ---------------------------------------------------------------------------

def test_extraction_agent_raises_for_missing_file():
    ctx = PipelineContext(
        pdf_path=Path("nonexistent_totally_fake_file_xyz.pdf"),
        template_path=Path("docs/Medical Summary.docx"),
        output_path=Path("output/test_summary.docx"),
    )
    agent = ExtractionAgent()
    with pytest.raises(Exception):
        agent.run(ctx)


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path: Path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content roundtrip")

    pages = [PageContent(page_number=1, text="Hello world test page")]
    doc = PDFDocument(path=pdf_path, pages=pages)
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    save_cache(pdf_path, h, doc, quality_score=0.95)
    result = load_cache(pdf_path)

    assert result is not None
    loaded_doc, loaded_hash, quality = result
    assert loaded_hash == h
    assert loaded_doc.total_pages == 1
    assert loaded_doc.pages[0].text == "Hello world test page"
    assert quality == 0.95


def test_cache_miss_returns_none(tmp_path: Path):
    pdf_path = tmp_path / "uncached.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 this file was never cached")
    assert load_cache(pdf_path) is None


def test_cache_hit_skips_extraction(tmp_path: Path, base_context: PipelineContext):
    """ExtractionAgent must load from cache and set cache_hit=True."""
    pages = [PageContent(page_number=1, text="Cached page content here " * 5)]
    doc = PDFDocument(path=base_context.pdf_path, pages=pages)
    h = hashlib.sha256(base_context.pdf_path.read_bytes()).hexdigest()
    save_cache(base_context.pdf_path, h, doc, quality_score=1.0)

    agent = ExtractionAgent()
    result = agent.run(base_context)

    assert result.cache_hit is True
    assert result.pdf_document is not None
    assert result.pdf_document.total_pages == 1
    assert result.extraction_quality == 1.0


# ---------------------------------------------------------------------------
# pypdfium2 fallback
# ---------------------------------------------------------------------------

def test_pypdfium2_fallback_used_when_quality_low(tmp_path: Path):
    """When pdfplumber quality < threshold, pypdfium2 is tried as fallback."""
    pdf_path = tmp_path / "low_quality.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")

    low_quality_doc = PDFDocument(path=pdf_path, pages=[PageContent(page_number=1, text="")])
    good_quality_doc = PDFDocument(
        path=pdf_path,
        pages=[PageContent(page_number=1, text="x" * 200)],
    )

    with (
        patch.object(ExtractionAgent, "_extract_pdfplumber", return_value=(low_quality_doc, 0.0)),
        patch.object(ExtractionAgent, "_extract_pypdfium2", return_value=(good_quality_doc, 1.0)),
        patch("medical_summary_builder.agents.extraction_agent.save_cache"),
        patch("medical_summary_builder.agents.extraction_agent.load_cache", return_value=None),
        patch("medical_summary_builder.cache._pdf_hash", return_value="abc123"),
    ):
        ctx = PipelineContext(
            pdf_path=pdf_path,
            template_path=Path("docs/Medical Summary.docx"),
            output_path=tmp_path / "out.docx",
        )
        agent = ExtractionAgent()
        result = agent.run(ctx)

    assert result.extraction_quality == 1.0
    assert result.pdf_document.pages[0].text == "x" * 200


def test_pypdfium2_not_used_when_quality_high(tmp_path: Path):
    """When pdfplumber quality >= threshold, pypdfium2 must NOT be called."""
    pdf_path = tmp_path / "high_quality.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")

    good_doc = PDFDocument(
        path=pdf_path,
        pages=[PageContent(page_number=1, text="x" * 200)],
    )

    with (
        patch.object(ExtractionAgent, "_extract_pdfplumber", return_value=(good_doc, 1.0)),
        patch.object(ExtractionAgent, "_extract_pypdfium2") as mock_pypdfium,
        patch("medical_summary_builder.agents.extraction_agent.save_cache"),
        patch("medical_summary_builder.agents.extraction_agent.load_cache", return_value=None),
        patch("medical_summary_builder.cache._pdf_hash", return_value="abc123"),
    ):
        ctx = PipelineContext(
            pdf_path=pdf_path,
            template_path=Path("docs/Medical Summary.docx"),
            output_path=tmp_path / "out.docx",
        )
        agent = ExtractionAgent()
        agent.run(ctx)

    mock_pypdfium.assert_not_called()
