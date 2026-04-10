"""AnalysisAgent — LLM-powered structured extraction from PDF text.

Chunked extraction strategy (replaces the old single-call approach):

1. Extract claimant demographics from the first DEMOGRAPHICS_PAGES pages only.
2. Auto-detect F-section boundaries by scanning for "1 of N: XF:" cover pages.
3. Call the LLM once per F section to extract every individual clinical encounter.
4. Merge, deduplicate by (date, provider), and sort all events chronologically.

Fallback: if no F-section markers are found, send the full document in one call
(legacy behaviour, used for non-standard PDF layouts and unit tests).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseAgent
from ..pipeline import ClaimantInfo, MedicalEvent, PDFDocument, PipelineContext

AI_BUILDERS_BASE_URL = "https://space.ai-builders.com/backend/v1"

DEMOGRAPHICS_PAGES = 30   # first N pages used for demographic extraction

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEMOGRAPHICS_SYSTEM_PROMPT = """
You are a medical records analyst. Extract structured claimant demographics
from the opening pages of a Social Security disability case file.

Return ONLY valid JSON — no markdown fences, no explanation:
{
  "name": "Full name",
  "ssn": "XXX-XX-XXXX",
  "title": "T2 / T16 / etc.",
  "dli": "MM/DD/YYYY or empty string",
  "aod": "MM/DD/YYYY",
  "dob": "MM/DD/YYYY",
  "age_at_aod": "integer string",
  "last_grade": "last grade of school completed",
  "special_ed": "Yes or No",
  "alleged_impairments": ["list", "of", "impairments"]
}

Field extraction notes:
- dli: Look for labels "DLI", "Date Last Insured", "Last Insured", or "Last Insured: Application:".
  The date immediately following any of those labels is the DLI value.
- title: Look for "Claim type:", "T2", "T16", "DI", "DIB", "SSI", etc.
- aod: Look for "Alleged onset:", "AOD:", or "Alleged Onset Date".
- age_at_aod: extract the numeric age at alleged onset (e.g. "57" from "57 years 1 month").
- last_grade: extract the numeric grade or description (e.g. "12" from "EDUCATION: 12th").
- special_ed: "Yes" or "No" based on any mention of special education.

If a field is not found, use an empty string (or empty list for impairments).
""".strip()

EVENTS_SYSTEM_PROMPT = """
You are a medical records analyst. Extract EVERY individual clinical encounter
from this section of medical records.

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {
    "date": "MM/DD/YYYY",
    "provider": "Facility or provider name",
    "physician": "Treating physician or clinician name (e.g. 'Dr. Smith' or 'John Smith, MD'), empty string if not stated",
    "reason": "Brief reason for visit / key finding (max 10 words)",
    "ref": "Pg N"
  }
]

Rules:
- Extract EVERY visit, appointment, procedure, lab review, admission, and discharge.
- Use the nearest preceding "--- Page N ---" marker for the ref field.
- Sort chronologically by date.
- Omit purely administrative entries (authorization letters, billing forms, notices).
- If no clinical encounters are found, return an empty array [].
""".strip()

CUSTOM_LAYOUT_SYSTEM_PROMPT = """
You are a medical records analyst. Given structured claimant data (JSON) and a
custom column instruction, reformat the medical timeline into a table that matches
the requested columns.

Return ONLY a JSON array of row objects. Each key must match one of the requested
column names exactly (normalised to lowercase with underscores).

Example instruction: "5 columns: Date, Facility, Physician, Summary, Ref"
Example row: {"date": "...", "facility": "...", "physician": "...", "summary": "...", "ref": "..."}

If a column value cannot be determined from the data, use an empty string.
Do NOT wrap output in markdown code fences.
""".strip()

# Legacy prompt kept for the fallback path (no F-section markers detected)
EXTRACTION_SYSTEM_PROMPT = """
You are a medical records analyst. Given the full text of a disability case file
(with page markers like "--- Page N ---"), extract structured information precisely.

Return ONLY valid JSON matching this schema:
{
  "name": "Claimant full name",
  "ssn": "SSN in XXX-XX-XXXX format",
  "title": "Title designation (T2 / T16 / etc.)",
  "dli": "Date Last Insured (MM/DD/YYYY)",
  "aod": "Alleged Onset Date (MM/DD/YYYY)",
  "dob": "Date of Birth (MM/DD/YYYY)",
  "age_at_aod": "Age at AOD as integer string",
  "last_grade": "Last grade of school completed",
  "special_ed": "Yes or No",
  "alleged_impairments": ["list", "of", "impairments"],
  "medical_events": [
    {
      "date": "MM/DD/YYYY",
      "provider": "Facility or provider name",
      "physician": "Treating physician or clinician name, empty string if not stated",
      "reason": "Brief reason for visit / key findings (max 10 words)",
      "ref": "Pg N"
    }
  ]
}

Rules:
- medical_events must be sorted chronologically.
- Use the page number in the nearest preceding "--- Page N ---" marker for ref.
- If a field is unknown, use an empty string.
- Do NOT wrap output in markdown code fences.
""".strip()

_DEMOGRAPHIC_FIELDS = frozenset({
    "name", "ssn", "title", "dli", "aod", "dob",
    "age_at_aod", "last_grade", "special_ed", "alleged_impairments",
})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseAgent):
    name = "Analysis Agent"

    def _run(self, context: PipelineContext) -> PipelineContext:
        if context.pdf_document is None:
            raise RuntimeError("AnalysisAgent requires pdf_document in context.")

        console.print(
            f"Sending [bold]{context.pdf_document.total_pages}[/bold] pages to "
            f"[cyan]{context.model}[/cyan] for structured extraction…"
        )

        claimant = _extract_claimant_info(
            context.pdf_document,
            model=context.model,
        )
        context.claimant_info = claimant

        console.print(Panel(
            f"[bold]Claimant:[/bold] {claimant.name or 'Unknown'}\n"
            f"[bold]SSN:[/bold] {claimant.ssn}\n"
            f"[bold]DOB:[/bold] {claimant.dob}  |  "
            f"[bold]AOD:[/bold] {claimant.aod}\n"
            f"[bold]Impairments:[/bold] {', '.join(claimant.alleged_impairments) or '—'}\n"
            f"[bold]Medical events found:[/bold] {len(claimant.medical_events)}",
            title="Extracted Claimant Info",
            border_style="green",
        ))

        return context


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_custom_layout(
    claimant: ClaimantInfo,
    layout_instruction: str,
    model: str,
) -> list[dict[str, str]]:
    """Re-format the medical timeline per the user's column instruction."""
    raw = _call_llm(
        system=CUSTOM_LAYOUT_SYSTEM_PROMPT,
        user=(
            f"Column instruction: {layout_instruction}\n\n"
            f"Claimant data (JSON):\n{claimant.model_dump_json(indent=2)}"
        ),
        model=model,
    )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _find_medical_sections(doc: PDFDocument) -> list[tuple[str, int, int, str]]:
    """Scan document pages for F-section cover pages.

    A cover page starts with "1 of N: XF: <type> Src: <source>" which marks
    the first page of each medical-record exhibit.

    Returns list of (section_id, start_page, end_page, source_name).
    """
    sections: list[tuple[str, int, int, str]] = []

    for page in doc.pages:
        text = page.text.strip()
        m = re.search(r'\b1\s+of\s+(\d+):\s*(\d+F):', text, re.IGNORECASE)
        if not m:
            continue

        total_pages = int(m.group(1))
        section_id = m.group(2).upper()
        start = page.page_number
        end = min(start + total_pages - 1, doc.total_pages)

        # Extract source name (text after "Src:")
        src_match = re.search(r'Src:\s*([^\n]+)', text, re.IGNORECASE)
        source = src_match.group(1).strip() if src_match else section_id
        # Strip trailing date ranges like "01/16/2023 To:..." or "09/15/2022 - ..."
        source = re.sub(r'\s+\d{2}/\d{2}/\d{4}.*$', '', source).strip()

        sections.append((section_id, start, end, source))

    return sections


# ---------------------------------------------------------------------------
# Extraction orchestration
# ---------------------------------------------------------------------------

def _extract_claimant_info(pdf_document: PDFDocument, model: str) -> ClaimantInfo:
    """Orchestrate demographics extraction + per-F-section event extraction."""
    # Step 1 — Demographics from opening pages only
    demo_pages = min(DEMOGRAPHICS_PAGES, pdf_document.total_pages)
    demo_text = "\n\n".join(
        f"--- Page {p.page_number} ---\n{p.text}"
        for p in pdf_document.pages[:demo_pages]
    )
    raw_demo = _call_llm(
        system=DEMOGRAPHICS_SYSTEM_PROMPT,
        user=f"Case file opening pages:\n\n{demo_text}",
        model=model,
    )
    demo_data = json.loads(raw_demo)
    # Keep only demographic fields — discard any stray medical_events the LLM added
    demographics = {k: v for k, v in demo_data.items() if k in _DEMOGRAPHIC_FIELDS}

    # Step 2 — Detect F-section boundaries
    sections = _find_medical_sections(pdf_document)

    if not sections:
        console.print(
            "[yellow]No F-section markers found — falling back to "
            "full-document extraction.[/yellow]"
        )
        return _fallback_full_extraction(pdf_document.full_text, model)

    console.print(
        f"Found [bold]{len(sections)}[/bold] medical record sections to extract."
    )

    # Step 3 — Extract events from each section independently
    all_events: list[MedicalEvent] = []
    for section_id, start, end, source in sections:
        section_text = "\n\n".join(
            f"--- Page {n} ---\n{pdf_document.get_page_text(n)}"
            for n in range(start, end + 1)
        )
        console.print(
            f"  [{section_id}] {source} "
            f"(Pg {start}–{end}, {end - start + 1} pages)…"
        )
        events = _extract_events_from_section(section_text, section_id, source, model)
        all_events.extend(events)
        console.print(f"    → {len(events)} events extracted")

    # Step 4 — Deduplicate and sort chronologically
    all_events = _deduplicate_events(all_events)
    all_events.sort(key=lambda e: (_date_sort_key(e.date), e.provider))

    return ClaimantInfo(**demographics, medical_events=all_events)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _extract_events_from_section(
    section_text: str,
    section_id: str,
    source: str,
    model: str,
) -> list[MedicalEvent]:
    """Extract all clinical events from one F section via LLM."""
    try:
        raw = _call_llm(
            system=EVENTS_SYSTEM_PROMPT,
            user=f"Medical records section {section_id} — {source}:\n\n{section_text}",
            model=model,
        )
        data = json.loads(raw)
        return [MedicalEvent(**e) for e in data]
    except Exception as exc:
        logger.warning(
            "Event extraction failed for %s: %s — skipping section", section_id, exc
        )
        return []


def _date_sort_key(date_str: str) -> tuple[int, int, int]:
    """Parse MM/DD/YYYY into a (year, month, day) tuple for chronological sorting.

    Falls back to (9999, 12, 31) so malformed dates sort last instead of crashing.
    """
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    return (9999, 12, 31)


def _deduplicate_events(events: list[MedicalEvent]) -> list[MedicalEvent]:
    """Remove duplicate events sharing the same (date, provider) pair."""
    seen: set[tuple[str, str]] = set()
    unique: list[MedicalEvent] = []
    for event in events:
        key = (event.date.strip(), event.provider.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fallback_full_extraction(pdf_text: str, model: str) -> ClaimantInfo:
    """Single-call extraction for documents without F-section markers."""
    raw = _call_llm(
        system=EXTRACTION_SYSTEM_PROMPT,
        user=f"Here is the medical case file text:\n\n{pdf_text}",
        model=model,
    )
    data = json.loads(raw)
    return ClaimantInfo(**data)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, model: str) -> str:
    token = os.environ.get("AI_BUILDER_TOKEN")
    if not token:
        raise EnvironmentError(
            "AI_BUILDER_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    client = OpenAI(base_url=AI_BUILDERS_BASE_URL, api_key=token)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
