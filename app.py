"""FastAPI web server for Medical Summary Builder."""

from __future__ import annotations

import logging
import logging.config
import os
import sys
import threading
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── logging setup ──────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
    },
    "loggers": {
        "app": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # Suppress noisy pipeline debug logs in web context
        "medical_summary_builder": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
})

logger = logging.getLogger("app")

# Make the pipeline importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medical_summary_builder.pipeline import Pipeline, PipelineContext
from medical_summary_builder.agents import (
    ExtractionAgent,
    AnalysisAgent,
    ValidationAgent,
    ReportAgent,
)

app = FastAPI(title="Medical Summary Builder")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEFAULT_TEMPLATE = BASE_DIR / "docs" / "summary_template.docx"
WORK_DIR = Path(tempfile.gettempdir()) / "medsummary"
OUTPUT_DIR = WORK_DIR / "output"
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── in-memory job store ────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

PIPELINE_STAGES = [
    "Extraction Agent: parsing PDF…",
    "Analysis Agent: extracting demographics and medical events…",
    "Validation Agent: fact-checking visit rows…",
    "Report Agent: generating Word document…",
    "Finalising output…",
]


def _run_pipeline(
    job_id: str,
    pdf_path: Path,
    template_path: Path,
    output_path: Path,
    model: str,
    layout: str | None,
) -> None:
    """Execute the pipeline in a background thread."""
    t0 = time.perf_counter()
    logger.info(
        "[job:%s] started | model=%s layout=%s pdf=%s",
        job_id[:8], model, repr(layout), pdf_path.name,
    )

    try:
        jobs[job_id].update(status="running", message=PIPELINE_STAGES[0], stage=0)

        context = PipelineContext(
            pdf_path=pdf_path,
            template_path=template_path,
            output_path=output_path,
            model=model,
            layout_instruction=layout,
        )

        class ProgressPipeline(Pipeline):
            def run(self, ctx: PipelineContext) -> PipelineContext:
                for i, agent in enumerate(self.agents):
                    msg = PIPELINE_STAGES[min(i, len(PIPELINE_STAGES) - 1)]
                    jobs[job_id]["message"] = msg
                    jobs[job_id]["stage"] = i
                    logger.info("[job:%s] stage %d — %s", job_id[:8], i, msg)
                    ctx = agent.run(ctx)
                return ctx

        pipeline = ProgressPipeline(agents=[
            ExtractionAgent(),
            AnalysisAgent(),
            ValidationAgent(),
            ReportAgent(),
        ])

        final = pipeline.run(context)

        elapsed = time.perf_counter() - t0
        logger.info(
            "[job:%s] done in %.1fs | issues=%d completion_through=%s",
            job_id[:8], elapsed,
            len(final.validation_issues or []),
            getattr(final, "completion_through", ""),
        )
        jobs[job_id].update(
            status="done",
            message="Summary generated successfully!",
            stage=4,
            output_path=str(final.report_path),
            validation_issues=final.validation_issues or [],
            completion_through=getattr(final, "completion_through", ""),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("[job:%s] failed after %.1fs — %s", job_id[:8], elapsed, exc, exc_info=True)
        jobs[job_id].update(status="error", message=str(exc))

    finally:
        try:
            pdf_path.unlink(missing_ok=True)
            if template_path != DEFAULT_TEMPLATE:
                template_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/summarize")
async def summarize(
    pdf: UploadFile = File(...),
    template: Optional[UploadFile] = File(None),
    model: str = Form("gpt-5"),
    layout: Optional[str] = Form(None),
):
    if not os.getenv("AI_BUILDER_TOKEN"):
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN not configured on server.")

    job_id = str(uuid.uuid4())

    # Save uploaded PDF
    pdf_path = WORK_DIR / f"{job_id}_input.pdf"
    pdf_path.write_bytes(await pdf.read())

    # Template: use uploaded one or fall back to bundled default
    if template and template.filename:
        tpl_path = WORK_DIR / f"{job_id}_template.docx"
        tpl_path.write_bytes(await template.read())
    else:
        tpl_path = DEFAULT_TEMPLATE

    output_path = OUTPUT_DIR / f"summary_{job_id}.docx"
    layout_clean: str | None = layout.strip() if layout and layout.strip() else None

    jobs[job_id] = {
        "status": "pending",
        "message": "Job queued…",
        "stage": -1,
        "output_path": None,
        "validation_issues": [],
        "completion_through": "",
        "created_at": datetime.utcnow().isoformat(),
    }

    logger.info(
        "[job:%s] queued | pdf=%s tpl=%s model=%s layout=%s",
        job_id[:8], pdf.filename, tpl_path.name, model, repr(layout_clean),
    )

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, pdf_path, tpl_path, output_path, model, layout_clean),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "stage": job.get("stage", -1),
        "has_output": job.get("output_path") is not None,
        "validation_issues": job.get("validation_issues", []),
        "completion_through": job.get("completion_through", ""),
    }


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not complete yet")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=output_path,
        filename=f"medical_summary_{job_id[:8]}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
