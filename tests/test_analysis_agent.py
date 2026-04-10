"""Tests for AnalysisAgent and related LLM helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from medical_summary_builder.agents.analysis_agent import (
    AnalysisAgent,
    apply_custom_layout,
    _extract_claimant_info,
    _find_medical_sections,
    _deduplicate_events,
    _call_llm,
)
from medical_summary_builder.pipeline import (
    ClaimantInfo, MedicalEvent, PageContent, PDFDocument, PipelineContext,
)


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

DEMOGRAPHICS_JSON = json.dumps({
    "name": "Jane Smith",
    "ssn": "987-65-4321",
    "title": "T16",
    "dli": "06/30/2022",
    "aod": "01/01/2020",
    "dob": "05/20/1975",
    "age_at_aod": "44",
    "current_age": "50",
    "last_grade": "10",
    "special_ed": "No",
    "alleged_impairments": ["Depression", "Anxiety"],
})

EVENTS_JSON = json.dumps([
    {
        "date": "01/01/2020",
        "provider": "General Hospital",
        "reason": "Initial psychiatric evaluation",
        "ref": "Pg 5",
    }
])

# Combined (legacy / fallback format)
VALID_CLAIMANT_JSON = json.dumps({
    "name": "Jane Smith",
    "ssn": "987-65-4321",
    "title": "T16",
    "dli": "06/30/2022",
    "aod": "01/01/2020",
    "dob": "05/20/1975",
    "age_at_aod": "44",
    "current_age": "50",
    "last_grade": "10",
    "special_ed": "No",
    "alleged_impairments": ["Depression", "Anxiety"],
    "medical_events": [
        {
            "date": "01/01/2020",
            "provider": "General Hospital",
            "reason": "Initial psychiatric evaluation",
            "ref": "Pg 5",
        }
    ],
})


@pytest.fixture
def simple_doc(tmp_path: Path) -> PDFDocument:
    """A minimal 3-page PDFDocument with no F-section markers (triggers fallback)."""
    return PDFDocument(
        path=tmp_path / "fake.pdf",
        pages=[
            PageContent(page_number=1, text="John Doe SSN 123-45-6789 DOB 01/15/1960"),
            PageContent(page_number=2, text="AOD 03/01/2018 DLI 12/31/2023"),
            PageContent(page_number=3, text="City Medical Center lumbar pain"),
        ],
    )


@pytest.fixture
def doc_with_f_sections(tmp_path: Path) -> PDFDocument:
    """A PDFDocument containing two F-section cover pages."""
    return PDFDocument(
        path=tmp_path / "fake_f.pdf",
        pages=[
            PageContent(page_number=1, text="Claimant: Jane Smith SSN: 987-65-4321"),
            PageContent(
                page_number=2,
                text="1 of 2: 1F: Office Treatment Records - OFFCREC Src: GENERAL HOSPITAL 01/01/2020",
            ),
            PageContent(page_number=3, text="Visit 01/01/2020 chief complaint depression"),
            PageContent(
                page_number=4,
                text="1 of 1: 2F: Hospital Records - HOSPITAL Src: CITY CLINIC 02/01/2020",
            ),
            PageContent(page_number=5, text="Admission 02/01/2020 follow-up"),
        ],
    )


# ---------------------------------------------------------------------------
# _call_llm
# ---------------------------------------------------------------------------

class TestCallLlm:
    def test_raises_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("AI_BUILDER_TOKEN", raising=False)
        with pytest.raises(EnvironmentError, match="AI_BUILDER_TOKEN"):
            _call_llm(system="sys", user="user", model="grok-4-fast")

    def test_returns_content_string(self, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  hello world  "
        with patch("medical_summary_builder.agents.analysis_agent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = _call_llm(system="sys", user="user", model="grok-4-fast")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# _find_medical_sections
# ---------------------------------------------------------------------------

class TestFindMedicalSections:
    def test_detects_f_section_cover_pages(self, doc_with_f_sections):
        sections = _find_medical_sections(doc_with_f_sections)
        assert len(sections) == 2

    def test_returns_correct_ids(self, doc_with_f_sections):
        sections = _find_medical_sections(doc_with_f_sections)
        ids = [s[0] for s in sections]
        assert "1F" in ids
        assert "2F" in ids

    def test_returns_correct_page_range(self, doc_with_f_sections):
        sections = _find_medical_sections(doc_with_f_sections)
        # 1F: starts at page 2, "1 of 2" → ends at page 3
        one_f = next(s for s in sections if s[0] == "1F")
        assert one_f[1] == 2   # start
        assert one_f[2] == 3   # end

    def test_empty_for_no_markers(self, simple_doc):
        assert _find_medical_sections(simple_doc) == []

    def test_extracts_source_name(self, doc_with_f_sections):
        sections = _find_medical_sections(doc_with_f_sections)
        one_f = next(s for s in sections if s[0] == "1F")
        assert "GENERAL HOSPITAL" in one_f[3]


# ---------------------------------------------------------------------------
# _deduplicate_events
# ---------------------------------------------------------------------------

class TestDeduplicateEvents:
    def test_removes_exact_duplicates(self):
        e1 = MedicalEvent(date="01/01/2020", provider="Hospital A", reason="visit", ref="Pg 1")
        e2 = MedicalEvent(date="01/01/2020", provider="Hospital A", reason="visit copy", ref="Pg 2")
        result = _deduplicate_events([e1, e2])
        assert len(result) == 1

    def test_case_insensitive_provider(self):
        e1 = MedicalEvent(date="01/01/2020", provider="hospital a", reason="x", ref="Pg 1")
        e2 = MedicalEvent(date="01/01/2020", provider="HOSPITAL A", reason="x", ref="Pg 1")
        result = _deduplicate_events([e1, e2])
        assert len(result) == 1

    def test_preserves_different_events(self):
        e1 = MedicalEvent(date="01/01/2020", provider="Hospital A", reason="x", ref="Pg 1")
        e2 = MedicalEvent(date="02/01/2020", provider="Hospital A", reason="y", ref="Pg 2")
        result = _deduplicate_events([e1, e2])
        assert len(result) == 2

    def test_empty_input(self):
        assert _deduplicate_events([]) == []


# ---------------------------------------------------------------------------
# _extract_claimant_info — fallback path (no F sections)
# ---------------------------------------------------------------------------

class TestExtractClaimantInfoFallback:
    def test_parses_valid_json(self, monkeypatch, simple_doc):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        # simple_doc has no F sections → demographics call + fallback call
        with patch(
            "medical_summary_builder.agents.analysis_agent._call_llm",
            return_value=VALID_CLAIMANT_JSON,
        ):
            claimant = _extract_claimant_info(simple_doc, model="grok-4-fast")

        assert isinstance(claimant, ClaimantInfo)
        assert claimant.name == "Jane Smith"
        assert claimant.ssn == "987-65-4321"
        assert len(claimant.medical_events) == 1
        assert claimant.medical_events[0].provider == "General Hospital"

    def test_raises_on_invalid_json(self, monkeypatch, simple_doc):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        with (
            patch(
                "medical_summary_builder.agents.analysis_agent._call_llm",
                return_value="not valid json!!!",
            ),
            pytest.raises(Exception),
        ):
            _extract_claimant_info(simple_doc, model="grok-4-fast")

    def test_empty_fields_become_defaults(self, monkeypatch, simple_doc):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        minimal_json = json.dumps({
            "name": "", "ssn": "", "title": "", "dli": "", "aod": "",
            "dob": "", "age_at_aod": "", "current_age": "", "last_grade": "",
            "special_ed": "", "alleged_impairments": [], "medical_events": [],
        })
        with patch(
            "medical_summary_builder.agents.analysis_agent._call_llm",
            return_value=minimal_json,
        ):
            claimant = _extract_claimant_info(simple_doc, model="grok-4-fast")
        assert claimant.name == ""
        assert claimant.medical_events == []


# ---------------------------------------------------------------------------
# _extract_claimant_info — chunked F-section path
# ---------------------------------------------------------------------------

class TestExtractClaimantInfoChunked:
    def test_calls_llm_per_section(self, monkeypatch, doc_with_f_sections):
        """Expect one demographics call + one call per F section (2 sections = 3 total)."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        call_count = {"n": 0}

        def fake_llm(system, user, model):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return DEMOGRAPHICS_JSON          # demographics call
            return EVENTS_JSON                    # per-section event calls

        with patch("medical_summary_builder.agents.analysis_agent._call_llm", side_effect=fake_llm):
            claimant = _extract_claimant_info(doc_with_f_sections, model="grok-4-fast")

        # 1 demographics + 2 F sections = 3 calls
        assert call_count["n"] == 3

    def test_merges_events_from_all_sections(self, monkeypatch, doc_with_f_sections):
        """Events from different sections must all appear in the result."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        section_events = [
            json.dumps([{"date": "01/01/2020", "provider": "Hospital A", "reason": "visit A", "ref": "Pg 3"}]),
            json.dumps([{"date": "02/01/2020", "provider": "City Clinic", "reason": "visit B", "ref": "Pg 5"}]),
        ]
        call_count = {"n": 0}

        def fake_llm(system, user, model):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return DEMOGRAPHICS_JSON
            idx = call_count["n"] - 2          # 0-based section index
            return section_events[min(idx, len(section_events) - 1)]

        with patch("medical_summary_builder.agents.analysis_agent._call_llm", side_effect=fake_llm):
            claimant = _extract_claimant_info(doc_with_f_sections, model="grok-4-fast")

        providers = {e.provider for e in claimant.medical_events}
        assert "Hospital A" in providers
        assert "City Clinic" in providers

    def test_deduplicates_across_sections(self, monkeypatch, doc_with_f_sections):
        """Same event appearing in two sections should only appear once."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        duplicate_event = json.dumps([
            {"date": "01/01/2020", "provider": "Hospital A", "reason": "visit", "ref": "Pg 3"}
        ])
        call_count = {"n": 0}

        def fake_llm(system, user, model):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return DEMOGRAPHICS_JSON
            return duplicate_event

        with patch("medical_summary_builder.agents.analysis_agent._call_llm", side_effect=fake_llm):
            claimant = _extract_claimant_info(doc_with_f_sections, model="grok-4-fast")

        assert len(claimant.medical_events) == 1

    def test_demographics_fields_populated(self, monkeypatch, doc_with_f_sections):
        """Demographics extracted from opening pages should be present in result."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        call_count = {"n": 0}

        def fake_llm(system, user, model):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return DEMOGRAPHICS_JSON
            return "[]"

        with patch("medical_summary_builder.agents.analysis_agent._call_llm", side_effect=fake_llm):
            claimant = _extract_claimant_info(doc_with_f_sections, model="grok-4-fast")

        assert claimant.name == "Jane Smith"
        assert claimant.ssn == "987-65-4321"
        assert claimant.aod == "01/01/2020"


# ---------------------------------------------------------------------------
# AnalysisAgent._run
# ---------------------------------------------------------------------------

class TestAnalysisAgent:
    def test_raises_without_pdf_document(self, base_context: PipelineContext):
        agent = AnalysisAgent()
        with pytest.raises(RuntimeError, match="pdf_document"):
            agent.run(base_context)

    def test_populates_claimant_info(self, context_with_doc: PipelineContext, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        with patch(
            "medical_summary_builder.agents.analysis_agent._call_llm",
            return_value=VALID_CLAIMANT_JSON,
        ):
            result = AnalysisAgent().run(context_with_doc)

        assert result.claimant_info is not None
        assert result.claimant_info.name == "Jane Smith"
        assert len(result.claimant_info.alleged_impairments) == 2

    def test_full_text_sent_in_fallback(self, context_with_doc: PipelineContext, monkeypatch):
        """When no F sections are present, full_text must appear in one of the LLM calls."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        captured_users: list[str] = []

        def fake_call_llm(system: str, user: str, model: str) -> str:
            captured_users.append(user)
            return VALID_CLAIMANT_JSON

        with patch(
            "medical_summary_builder.agents.analysis_agent._call_llm",
            side_effect=fake_call_llm,
        ):
            AnalysisAgent().run(context_with_doc)

        full_text = context_with_doc.pdf_document.full_text
        assert any(full_text in u for u in captured_users)


# ---------------------------------------------------------------------------
# apply_custom_layout
# ---------------------------------------------------------------------------

class TestApplyCustomLayout:
    def test_returns_list_of_dicts(self, sample_claimant, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        custom_rows = json.dumps([
            {"date": "03/01/2018", "facility": "City Medical Center", "summary": "Back pain", "ref": "Pg 3"}
        ])
        with patch("medical_summary_builder.agents.analysis_agent._call_llm", return_value=custom_rows):
            rows = apply_custom_layout(sample_claimant, "Date, Facility, Summary, Ref", "grok-4-fast")

        assert isinstance(rows, list)
        assert len(rows) == 1
        assert rows[0]["date"] == "03/01/2018"

    def test_raises_on_bad_json_from_llm(self, sample_claimant, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        with (
            patch("medical_summary_builder.agents.analysis_agent._call_llm", return_value="bad json"),
            pytest.raises(Exception),
        ):
            apply_custom_layout(sample_claimant, "Date, Facility", "grok-4-fast")
