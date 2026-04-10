"""ValidationAgent — self-correction and anti-hallucination check.

For each medical_event extracted by the AnalysisAgent:

  Step A  Fast fuzzy check (no LLM cost)
          Parse the page number from event.ref, fetch that page's text,
          and use rapidfuzz to test whether the provider name or a date
          fragment appears there.  Threshold: 60 → pass, else flag.

  Step B  Batched LLM self-correction (only for flagged events)
          Flagged events are grouped by page proximity: events whose
          ±CONTEXT_WINDOW page ranges overlap are sent in a single LLM
          call.  This dramatically reduces API round-trips when many
          events are flagged.  The LLM returns a corrected event JSON or
          "REMOVE" for each suspicious event.

The corrected ClaimantInfo replaces context.claimant_info.
"""

from __future__ import annotations

import json
import logging
import re

from rapidfuzz import fuzz
from rich.console import Console
from rich.table import Table

from .base import BaseAgent
from .analysis_agent import _call_llm, _date_sort_key
from ..pipeline import ClaimantInfo, MedicalEvent, PipelineContext

logger = logging.getLogger(__name__)
console = Console()

FUZZY_THRESHOLD = 60
CONTEXT_WINDOW = 3   # pages around the flagged event for LLM correction


BATCH_SELF_CORRECTION_SYSTEM_PROMPT = """
You are a medical records quality controller. Multiple medical events were extracted
from a PDF but their claimed page references may not match the page text.

You will receive:
1. A JSON array of suspicious events (0-indexed).
2. The text of all relevant pages covering these events.

For EACH event in order, return either:
- A corrected event JSON object  (if the event is supported by the page text, with the
  accurate "ref", e.g. "Pg 23")
- The string "REMOVE"  (if the event content cannot be found on any provided page)

Return ONLY a JSON array with exactly one entry per input event, in the same order.
No explanation. No markdown fences.
Example input:  [event0, event1, event2]
Example output: [{"date":"...","provider":"...","physician":"...","reason":"...","ref":"Pg 23"}, "REMOVE", {"date":"...","provider":"...","physician":"...","reason":"...","ref":"Pg 45"}]
""".strip()


class ValidationAgent(BaseAgent):
    name = "Validation Agent"

    def _run(self, context: PipelineContext) -> PipelineContext:
        if context.claimant_info is None or context.pdf_document is None:
            raise RuntimeError("ValidationAgent requires claimant_info and pdf_document.")

        claimant = context.claimant_info
        events = claimant.medical_events

        if not events:
            logger.info("No medical events to validate")
            console.print("[yellow]No medical events to validate.[/yellow]")
            context.validation_passed = True
            return context

        logger.info("Validating %d medical events", len(events))
        console.print(
            f"Validating [bold]{len(events)}[/bold] medical events against PDF pages…"
        )

        passed: list[MedicalEvent] = []
        flagged: list[MedicalEvent] = []
        issues: list[str] = []

        # Step A — fast fuzzy check
        for event in events:
            page_num = _parse_page_number(event.ref)
            if page_num is None:
                issues.append(f"Cannot parse ref '{event.ref}' for event: {event.provider}")
                flagged.append(event)
                continue

            page_text = context.pdf_document.get_page_text(page_num)
            if not page_text:
                issues.append(f"Pg {page_num} not found in document for event: {event.provider}")
                flagged.append(event)
                continue

            score = _fuzzy_score(event, page_text)
            if score >= FUZZY_THRESHOLD:
                passed.append(event)
            else:
                issues.append(
                    f"Low match ({score:.0f}/100) on Pg {page_num} "
                    f"for '{event.provider}' — {event.date}"
                )
                flagged.append(event)

        self._print_fuzzy_report(events, passed, flagged)

        # Step B — batched LLM self-correction for flagged events
        corrected: list[MedicalEvent] = list(passed)
        if flagged:
            batch_results = self._batch_llm_correct(flagged, context)
            for event, result in zip(flagged, batch_results):
                if result is not None:
                    corrected.append(result)
                    console.print(f"[cyan]Corrected:[/cyan] {event.provider} → {result.ref}")
                else:
                    issues.append(f"REMOVED (hallucination): {event.provider} {event.date}")
                    console.print(
                        f"[red]Removed hallucinated event:[/red] {event.provider} {event.date}"
                    )

        # Sort corrected events chronologically with proper MM/DD/YYYY parsing
        corrected.sort(key=lambda e: (_date_sort_key(e.date), e.provider))

        context.claimant_info = claimant.model_copy(update={"medical_events": corrected})
        context.validation_issues = issues
        context.validation_passed = len(flagged) == 0 or len(corrected) > 0

        removed = len(flagged) - (len(corrected) - len(passed))
        logger.info(
            "Validation complete: %d passed, %d flagged, %d removed, %d final",
            len(passed), len(flagged), removed, len(corrected),
        )
        for issue in issues:
            logger.debug("Validation issue: %s", issue)

        console.print(
            f"[bold]Validation complete:[/bold] "
            f"{len(passed)} passed, "
            f"{len(flagged)} flagged, "
            f"{removed} removed. "
            f"Final events: {len(corrected)}"
        )

        return context

    def _batch_llm_correct(
        self,
        flagged: list[MedicalEvent],
        context: PipelineContext,
    ) -> list[MedicalEvent | None]:
        """Correct flagged events using proximity-grouped batch LLM calls.

        Events whose ±CONTEXT_WINDOW page ranges overlap are sent together in a
        single LLM call.  Returns a result list in the same order as *flagged*.
        """
        groups = _group_events_by_proximity(flagged)
        console.print(
            f"  Batching [bold]{len(flagged)}[/bold] flagged event(s) into "
            f"[bold]{len(groups)}[/bold] LLM call(s)…"
        )

        results_by_id: dict[int, MedicalEvent | None] = {}
        for group in groups:
            group_results = self._correct_group(group, context)
            for event, result in zip(group, group_results):
                results_by_id[id(event)] = result

        return [results_by_id[id(e)] for e in flagged]

    def _correct_group(
        self,
        group: list[MedicalEvent],
        context: PipelineContext,
    ) -> list[MedicalEvent | None]:
        """Send one batch LLM call for a group of nearby flagged events."""
        doc = context.pdf_document

        # Union of all page windows for every event in the group
        page_set: set[int] = set()
        for event in group:
            page_num = _parse_page_number(event.ref) or 1
            for n in range(
                max(1, page_num - CONTEXT_WINDOW),
                min(doc.total_pages, page_num + CONTEXT_WINDOW) + 1,
            ):
                page_set.add(n)

        sorted_pages = sorted(page_set)
        pages_text = "\n\n".join(
            f"--- Page {n} ---\n{doc.get_page_text(n)}"
            for n in sorted_pages
        )
        events_json = json.dumps([e.model_dump() for e in group], indent=2)
        user_msg = (
            f"Suspicious events:\n{events_json}\n\n"
            f"Page texts (pages {sorted_pages[0]}–{sorted_pages[-1]}):\n{pages_text}"
        )

        try:
            raw = _call_llm(
                system=BATCH_SELF_CORRECTION_SYSTEM_PROMPT,
                user=user_msg,
                model=context.model,
            )
            # Strip optional markdown fences
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw)
            results_data: list = json.loads(raw)

            results: list[MedicalEvent | None] = []
            for i, item in enumerate(results_data[: len(group)]):
                if isinstance(item, str) and item.strip().upper() == "REMOVE":
                    results.append(None)
                elif isinstance(item, dict):
                    try:
                        results.append(MedicalEvent(**item))
                    except Exception:
                        results.append(group[i])  # keep original on parse error
                else:
                    results.append(group[i])

            # Pad if LLM returned fewer entries than expected
            while len(results) < len(group):
                results.append(group[len(results)])

            return results

        except Exception as exc:
            logger.warning(
                "Batch LLM correction failed: %s — keeping all originals in group", exc
            )
            return list(group)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_fuzzy_report(
        all_events: list[MedicalEvent],
        passed: list[MedicalEvent],
        flagged: list[MedicalEvent],
    ) -> None:
        table = Table(title="Fuzzy Validation Results", show_header=True)
        table.add_column("Date")
        table.add_column("Provider")
        table.add_column("Ref")
        table.add_column("Status")

        passed_set = {id(e) for e in passed}
        for event in all_events:
            status = (
                "[green]PASS[/green]" if id(event) in passed_set else "[yellow]FLAGGED[/yellow]"
            )
            table.add_row(event.date, event.provider, event.ref, status)

        console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_events_by_proximity(
    events: list[MedicalEvent],
) -> list[list[MedicalEvent]]:
    """Group events whose ±CONTEXT_WINDOW page ranges overlap into batches.

    Events with unparseable refs (page 0) are placed in their own single-event
    group so they do not artificially merge with real events.
    """
    if not events:
        return []

    def page_key(e: MedicalEvent) -> int:
        return _parse_page_number(e.ref) or 0

    sorted_events = sorted(events, key=page_key)
    groups: list[list[MedicalEvent]] = []
    current_group: list[MedicalEvent] = [sorted_events[0]]
    current_end = page_key(sorted_events[0]) + CONTEXT_WINDOW

    for event in sorted_events[1:]:
        pn = page_key(event)
        # Keep events with unknown page refs isolated
        if pn == 0 or pn - CONTEXT_WINDOW > current_end:
            groups.append(current_group)
            current_group = [event]
            current_end = pn + CONTEXT_WINDOW
        else:
            current_group.append(event)
            current_end = max(current_end, pn + CONTEXT_WINDOW)

    groups.append(current_group)
    return groups


def _parse_page_number(ref: str) -> int | None:
    """Extract integer page number from strings like 'Pg 19', 'Page 19', '19'."""
    match = re.search(r"\d+", ref or "")
    return int(match.group()) if match else None


def _fuzzy_score(event: MedicalEvent, page_text: str) -> float:
    """Return the best fuzzy match score between event fields and page_text."""
    page_lower = page_text.lower()
    candidates = [
        event.provider,
        # Use year + month from date for partial matching
        re.sub(r"[^0-9/]", "", event.date),
    ]
    return max(
        fuzz.partial_ratio(c.lower(), page_lower)
        for c in candidates
        if c.strip()
    )
