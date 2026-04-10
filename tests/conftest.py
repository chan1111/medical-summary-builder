"""Shared pytest fixtures for all agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_summary_builder.pipeline import (
    ClaimantInfo,
    MedicalEvent,
    PageContent,
    PDFDocument,
    PipelineContext,
)


# ---------------------------------------------------------------------------
# Minimal valid PDF bytes (a 1-page text-based PDF)
# ---------------------------------------------------------------------------

MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
441
%%EOF
"""


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A minimal valid PDF file written to a temp directory."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(MINIMAL_PDF_BYTES)
    return pdf_path


@pytest.fixture
def sample_pages() -> list[PageContent]:
    return [
        PageContent(page_number=1, text="John Doe SSN 123-45-6789 DOB 01/15/1960"),
        PageContent(page_number=2, text="AOD 03/01/2018 DLI 12/31/2023 chronic back pain"),
        PageContent(
            page_number=3,
            text="City Medical Center 03/01/2018 patient presents with lumbar pain",
        ),
    ]


@pytest.fixture
def sample_doc(fake_pdf: Path, sample_pages: list[PageContent]) -> PDFDocument:
    return PDFDocument(path=fake_pdf, pages=sample_pages)


@pytest.fixture
def sample_claimant(sample_doc: PDFDocument) -> ClaimantInfo:
    return ClaimantInfo(
        name="John Doe",
        ssn="123-45-6789",
        title="T2",
        dli="12/31/2023",
        aod="03/01/2018",
        dob="01/15/1960",
        age_at_aod="57",
        current_age="64",
        last_grade="12",
        special_ed="No",
        alleged_impairments=["Chronic back pain", "Hypertension"],
        medical_events=[
            MedicalEvent(
                date="03/01/2018",
                provider="City Medical Center",
                reason="Lumbar pain evaluation",
                ref="Pg 3",
            )
        ],
    )


@pytest.fixture
def base_context(fake_pdf: Path, tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        pdf_path=fake_pdf,
        template_path=Path("docs/Medical Summary.docx"),
        output_path=tmp_path / "output" / "summary.docx",
    )


@pytest.fixture
def context_with_doc(
    base_context: PipelineContext,
    sample_doc: PDFDocument,
) -> PipelineContext:
    base_context.pdf_document = sample_doc
    base_context.extraction_quality = 1.0
    return base_context


@pytest.fixture
def context_with_claimant(
    context_with_doc: PipelineContext,
    sample_claimant: ClaimantInfo,
) -> PipelineContext:
    context_with_doc.claimant_info = sample_claimant
    context_with_doc.validation_passed = True
    return context_with_doc
