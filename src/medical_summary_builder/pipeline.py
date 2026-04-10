"""Shared pipeline state and sequential runner."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models shared across agents
# ---------------------------------------------------------------------------

class PageContent(BaseModel):
    page_number: int
    text: str


class PDFDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    pages: list[PageContent] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All pages joined with clear page-break markers for LLM context."""
        return "\n\n".join(
            f"--- Page {p.page_number} ---\n{p.text}" for p in self.pages
        )

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def get_page_text(self, page_number: int) -> str:
        """Return text for a specific 1-indexed page, or empty string if missing."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return ""


class MedicalEvent(BaseModel):
    date: str = Field(description="Date of the medical event (MM/DD/YYYY or as written)")
    provider: str = Field(description="Facility or provider name")
    physician: str = Field(default="", description="Treating physician or clinician name, empty string if not stated")
    reason: str = Field(description="Reason for visit / summary of findings")
    ref: str = Field(description="Page number reference, e.g. 'Pg 19'")

    @field_validator("ref", mode="before")
    @classmethod
    def normalise_ref(cls, v: str) -> str:
        """Normalise any page reference to 'Pg <n>' (e.g. 'Page 5', 'p.5', '5' → 'Pg 5')."""
        if not v:
            return v
        m = re.search(r"\d+", str(v))
        return f"Pg {m.group()}" if m else v


class ClaimantInfo(BaseModel):
    name: str = ""
    ssn: str = ""
    title: str = ""
    dli: str = ""
    aod: str = ""
    dob: str = ""
    age_at_aod: str = ""
    current_age: str = ""
    last_grade: str = ""
    special_ed: str = ""
    alleged_impairments: list[str] = Field(default_factory=list)
    medical_events: list[MedicalEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compute_current_age(self) -> "ClaimantInfo":
        """Always derive current_age from dob when dob is parseable."""
        if not self.dob:
            return self
        from datetime import datetime
        born: date | None = None
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
            try:
                born = datetime.strptime(self.dob.strip(), fmt).date()
                break
            except ValueError:
                continue
        if born is None:
            return self
        today = date.today()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        self.current_age = str(age)
        return self


# ---------------------------------------------------------------------------
# Pipeline context — single object passed through all agents
# ---------------------------------------------------------------------------

class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Inputs
    pdf_path: Path
    template_path: Path
    output_path: Path
    model: str = "gpt-5"

    # Intent Agent output
    layout_instruction: str | None = None   # None = use template

    # Cache state
    pdf_hash: str = ""
    cache_hit: bool = False

    # Extraction Agent output
    pdf_document: PDFDocument | None = None
    extraction_quality: float = 0.0         # 0.0–1.0

    # Analysis Agent output
    claimant_info: ClaimantInfo | None = None
    medical_sections: list[dict] = Field(default_factory=list)
    # Each dict: {section_id, start_page, end_page, total_pages, source, date_range}

    # Validation Agent output
    validation_passed: bool = False
    validation_issues: list[str] = Field(default_factory=list)

    # Report Agent output
    report_path: Path | None = None
    completion_through: str = ""   # e.g. "F" when all F-section medical records are processed


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, agents: list[BaseAgent]) -> None:
        self.agents = agents

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info(
            "Pipeline starting — %d agent(s): %s",
            len(self.agents),
            ", ".join(a.name for a in self.agents),
        )
        for agent in self.agents:
            logger.info("--- Running %s ---", agent.name)
            context = agent.run(context)
            logger.info("%s done", agent.name)
        logger.info("Pipeline complete")
        return context
