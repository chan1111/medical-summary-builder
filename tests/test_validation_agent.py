"""Tests for ValidationAgent — fuzzy check and LLM self-correction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from medical_summary_builder.agents.validation_agent import (
    ValidationAgent,
    _fuzzy_score,
    _parse_page_number,
    FUZZY_THRESHOLD,
)
from medical_summary_builder.pipeline import (
    ClaimantInfo,
    MedicalEvent,
    PageContent,
    PDFDocument,
    PipelineContext,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestParsePageNumber:
    def test_pg_format(self):
        assert _parse_page_number("Pg 19") == 19

    def test_page_format(self):
        assert _parse_page_number("Page 3") == 3

    def test_bare_number(self):
        assert _parse_page_number("7") == 7

    def test_none_for_empty_string(self):
        assert _parse_page_number("") is None

    def test_none_for_no_digits(self):
        assert _parse_page_number("no digits here") is None

    def test_extracts_first_number(self):
        assert _parse_page_number("Pg 12 of 50") == 12


class TestFuzzyScore:
    def test_exact_provider_match_scores_high(self):
        event = MedicalEvent(
            date="01/01/2020",
            provider="City Medical Center",
            reason="check-up",
            ref="Pg 1",
        )
        page_text = "Visit to City Medical Center on January 1, 2020 for annual check-up."
        score = _fuzzy_score(event, page_text)
        assert score >= FUZZY_THRESHOLD

    def test_completely_wrong_provider_scores_low(self):
        event = MedicalEvent(
            date="01/01/2020",
            provider="XYZXYZXYZ Clinic",
            reason="check-up",
            ref="Pg 1",
        )
        page_text = "Patient presented to the emergency department with chest pain."
        score = _fuzzy_score(event, page_text)
        assert score < FUZZY_THRESHOLD

    def test_date_fragment_can_match(self):
        event = MedicalEvent(
            date="03/15/2021",
            provider="Unknown Clinic ZZZZZZ",
            reason="visit",
            ref="Pg 2",
        )
        page_text = "Appointment on 03/15/2021 for routine follow-up."
        score = _fuzzy_score(event, page_text)
        assert score >= FUZZY_THRESHOLD


# ---------------------------------------------------------------------------
# ValidationAgent — no events
# ---------------------------------------------------------------------------

class TestValidationAgentNoEvents:
    def test_passes_when_no_events(self, context_with_doc: PipelineContext):
        context_with_doc.claimant_info = ClaimantInfo(
            name="Test", medical_events=[]
        )
        agent = ValidationAgent()
        result = agent.run(context_with_doc)
        assert result.validation_passed is True

    def test_raises_without_claimant_info(self, context_with_doc: PipelineContext):
        context_with_doc.claimant_info = None
        with pytest.raises(RuntimeError, match="claimant_info"):
            ValidationAgent().run(context_with_doc)

    def test_raises_without_pdf_document(self, base_context: PipelineContext, sample_claimant):
        base_context.claimant_info = sample_claimant
        base_context.pdf_document = None
        with pytest.raises(RuntimeError, match="pdf_document"):
            ValidationAgent().run(base_context)


# ---------------------------------------------------------------------------
# ValidationAgent — fuzzy pass
# ---------------------------------------------------------------------------

class TestValidationFuzzyPass:
    def _make_context(self, tmp_path: Path) -> PipelineContext:
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"PDF")
        pages = [
            PageContent(
                page_number=3,
                text="City Medical Center evaluated patient on 03/01/2018 for lumbar pain",
            )
        ]
        doc = PDFDocument(path=fake_pdf, pages=pages)
        claimant = ClaimantInfo(
            name="John",
            medical_events=[
                MedicalEvent(
                    date="03/01/2018",
                    provider="City Medical Center",
                    reason="Lumbar pain evaluation",
                    ref="Pg 3",
                )
            ],
        )
        return PipelineContext(
            pdf_path=fake_pdf,
            template_path=Path("docs/Medical Summary.docx"),
            output_path=tmp_path / "out.docx",
            pdf_document=doc,
            claimant_info=claimant,
        )

    def test_fuzzy_pass_keeps_event(self, tmp_path):
        ctx = self._make_context(tmp_path)
        result = ValidationAgent().run(ctx)
        assert result.validation_passed is True
        assert len(result.claimant_info.medical_events) == 1
        assert result.claimant_info.medical_events[0].provider == "City Medical Center"


# ---------------------------------------------------------------------------
# ValidationAgent — fuzzy fail → LLM corrects
# ---------------------------------------------------------------------------

class TestValidationLlmCorrection:
    def _make_context_with_bad_ref(self, tmp_path: Path) -> PipelineContext:
        """Event ref points to a page that doesn't mention the provider."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"PDF")
        pages = [
            PageContent(page_number=1, text="Unrelated page about something else entirely"),
            PageContent(
                page_number=5,
                text="XYZXYZ Clinic saw the patient on 04/10/2021",
            ),
        ]
        doc = PDFDocument(path=fake_pdf, pages=pages)
        claimant = ClaimantInfo(
            name="Patient",
            medical_events=[
                MedicalEvent(
                    date="04/10/2021",
                    provider="XYZXYZ Clinic",
                    reason="Follow-up",
                    ref="Pg 1",  # wrong page — triggers flagging
                )
            ],
        )
        return PipelineContext(
            pdf_path=fake_pdf,
            template_path=Path("docs/Medical Summary.docx"),
            output_path=tmp_path / "out.docx",
            pdf_document=doc,
            claimant_info=claimant,
        )

    def test_llm_correction_updates_ref(self, tmp_path, monkeypatch):
        ctx = self._make_context_with_bad_ref(tmp_path)
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        corrected_event = json.dumps({
            "date": "04/10/2021",
            "provider": "XYZXYZ Clinic",
            "reason": "Follow-up",
            "ref": "Pg 5",
        })
        with patch("medical_summary_builder.agents.validation_agent._call_llm", return_value=corrected_event):
            result = ValidationAgent().run(ctx)

        assert len(result.claimant_info.medical_events) == 1
        assert result.claimant_info.medical_events[0].ref == "Pg 5"

    def test_llm_remove_eliminates_hallucinated_event(self, tmp_path, monkeypatch):
        ctx = self._make_context_with_bad_ref(tmp_path)
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        with patch("medical_summary_builder.agents.validation_agent._call_llm", return_value="REMOVE"):
            result = ValidationAgent().run(ctx)

        assert result.claimant_info.medical_events == []
        assert any("REMOVED" in issue for issue in result.validation_issues)

    def test_validation_issues_recorded_for_flagged_events(self, tmp_path, monkeypatch):
        ctx = self._make_context_with_bad_ref(tmp_path)
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        corrected = json.dumps({
            "date": "04/10/2021",
            "provider": "XYZXYZ Clinic",
            "reason": "Follow-up",
            "ref": "Pg 5",
        })
        with patch("medical_summary_builder.agents.validation_agent._call_llm", return_value=corrected):
            result = ValidationAgent().run(ctx)

        assert len(result.validation_issues) > 0


# ---------------------------------------------------------------------------
# ValidationAgent — unparseable ref
# ---------------------------------------------------------------------------

class TestValidationUnparseableRef:
    def test_unparseable_ref_flags_event(self, tmp_path, monkeypatch):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"PDF")
        doc = PDFDocument(
            path=fake_pdf,
            pages=[PageContent(page_number=1, text="some text")],
        )
        claimant = ClaimantInfo(
            name="Test",
            medical_events=[
                MedicalEvent(
                    date="01/01/2020",
                    provider="Some Clinic",
                    reason="Visit",
                    ref="unknown",  # no digits → unparseable
                )
            ],
        )
        ctx = PipelineContext(
            pdf_path=fake_pdf,
            template_path=Path("docs/Medical Summary.docx"),
            output_path=tmp_path / "out.docx",
            pdf_document=doc,
            claimant_info=claimant,
        )
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        corrected = json.dumps({
            "date": "01/01/2020",
            "provider": "Some Clinic",
            "reason": "Visit",
            "ref": "Pg 1",
        })
        with patch("medical_summary_builder.agents.validation_agent._call_llm", return_value=corrected):
            result = ValidationAgent().run(ctx)

        assert any("Cannot parse ref" in issue for issue in result.validation_issues)
