# Lessons Learned

This file tracks failures, unexpected behaviours, and hard-won insights encountered during development. Update it whenever something breaks or a better approach is discovered, so the same mistake is never made twice.

---

## Format

Each entry follows this structure:

```
### [YYYY-MM-DD] Short title of the issue
**Context:** What were we trying to do?
**What went wrong:** Description of the failure or bad assumption.
**Root cause:** Why did it happen?
**Fix / Lesson:** What was the correct approach?
**Tags:** #pdf #llm #docx #environment #prompt-engineering etc.
```

---

## Entries

### [2026-04-10] `tool.uv.dev-dependencies` is deprecated in uv 0.11+
**Context:** Setting up the initial `pyproject.toml` for the project.
**What went wrong:** Used `[tool.uv] dev-dependencies = [...]` which triggered a deprecation warning on every `uv sync`.
**Root cause:** uv moved to the PEP 735 standard `[dependency-groups]` table; the old `tool.uv` key is legacy.
**Fix / Lesson:** Use `[dependency-groups] dev = [...]` instead of `[tool.uv] dev-dependencies`.
**Tags:** #environment #uv

---

### [2026-04-10] Flat module layout doesn't scale for multi-agent pipelines
**Context:** Initial design put `pdf_extractor.py`, `llm_processor.py`, `report_generator.py` at the package root.
**What went wrong:** Adding new agents (ValidationAgent, IntentAgent) and shared state (PipelineContext) with cross-imports quickly becomes tangled and risks circular imports.
**Root cause:** Flat layout makes agent boundaries unclear and forces all models to be re-imported everywhere.
**Fix / Lesson:** Move all agents into an `agents/` sub-package; put shared data models in `pipeline.py`; put cache logic in `cache.py`. Each agent imports only from `..pipeline` and `..cache`.
**Tags:** #architecture #refactoring

---

### [2026-04-10] cache.py `parents[3]` pointed to drive root instead of project root
**Context:** ExtractionAgent saves/loads the PDF extraction cache via `cache.py`.
**What went wrong:** Cache files were written to `D:\cache\` (drive root) instead of `d:\medical_summary_builder\cache\`.
**Root cause:** `CACHE_DIR = Path(__file__).parents[3] / "cache"` — `__file__` is `src/medical_summary_builder/cache.py`, so `parents[2]` is the project root and `parents[3]` is the drive root `D:\`.
**Fix / Lesson:** Change `parents[3]` to `parents[2]`. Rule: count `parents` from the file's location — `[0]` = same dir, `[1]` = one up, `[2]` = project root for a `src/pkg/` layout.
**Tags:** #environment #cache

---

### [2026-04-10] Single LLM call on 192K-token document produces only a fraction of events
**Context:** AnalysisAgent sent the entire 504-page PDF (~192K tokens) in one LLM call to extract all medical events.
**What went wrong:** Only 6 events were returned from a document containing 10 F-section medical record exhibits spanning ~215 pages. The LLM suffered from the "lost in the middle" problem and silently skipped most visits.
**Root cause:** (1) 98K tokens of admin/legal noise diluted the medical content. (2) At 192K tokens, LLMs reliably extract only from the beginning and end of the context window. (3) Prompt didn't instruct exhaustive extraction of every individual encounter.
**Fix / Lesson:** Split extraction into chunks: detect F-section cover pages via the regex `1 of N: XF:`, call the LLM once per section (~3K–40K tokens each), then merge and deduplicate results. Event count increased from 6 → 21 with accurate per-section page refs.
**Tags:** #llm #prompt-engineering #context-window #chunking

---

### [2026-04-10] Word template must use `{{PLACEHOLDER}}` tokens and table header row
**Context:** ReportAgent fills the `.docx` template by replacing `{{NAME}}`, `{{SSN}}` etc. and calling `_fill_events_table()` to populate the timeline table.
**What went wrong:** The original template used blank underlines (`_________`) for fields and had no text in the table's first row, so `_populate_template()` never replaced anything and `_fill_events_table()` was never triggered (it checks for `"DATE"` in the header row).
**Root cause:** Template format mismatch — code expected `{{PLACEHOLDER}}` tokens, template had freeform blanks. Table detection relied on finding `"DATE"` in `table.rows[0]`, but the header labels were in a paragraph above the table, not in the table itself.
**Fix / Lesson:** Update the template with `{{PLACEHOLDER}}` tokens in every field paragraph, and put `DATE / PROVIDER / REASON / REF` as cell text in the table's first row. Back up the original `.docx` before modifying.
**Tags:** #docx #template #report-agent

---

### [2026-04-10] pypdfium2 API requires iterating the document object directly
**Context:** Implementing pypdfium2 fallback in ExtractionAgent.
**What went wrong:** Calling `pdf.pages` (like pdfplumber) raises AttributeError on a pypdfium2 PdfDocument.
**Root cause:** pypdfium2 follows a different convention — `PdfDocument` is iterable directly, not via a `.pages` attribute.
**Fix / Lesson:** Use `for i, page in enumerate(pdf, start=1)` directly on the `PdfDocument` object.
**Tags:** #pdf #pypdfium2
