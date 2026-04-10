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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

JOB_TTL_SECONDS = 2 * 60 * 60       # completed/errored jobs kept for 2 hours
UPLOAD_TTL_SECONDS = 15 * 60        # stale upload sessions expire after 15 minutes
CLEANUP_INTERVAL_SECONDS = 5 * 60   # run cleanup every 5 minutes

_stop_cleanup = threading.Event()


def _cleanup_loop() -> None:
    """Background thread: evict expired jobs and stale upload sessions."""
    while not _stop_cleanup.wait(timeout=CLEANUP_INTERVAL_SECONDS):
        now = datetime.now(timezone.utc)

        # --- clean up finished/errored jobs older than JOB_TTL_SECONDS ---
        expired_jobs = [
            jid for jid, j in list(jobs.items())
            if j.get("status") in ("done", "error")
            and (now - datetime.fromisoformat(j["created_at"])).total_seconds() > JOB_TTL_SECONDS
        ]
        for jid in expired_jobs:
            job = jobs.pop(jid, None)
            if job and job.get("output_path"):
                try:
                    Path(job["output_path"]).unlink(missing_ok=True)
                except Exception:
                    pass
        if expired_jobs:
            logger.info("Cleanup: removed %d expired job(s)", len(expired_jobs))

        # --- clean up stale upload sessions older than UPLOAD_TTL_SECONDS ---
        expired_uploads = [
            uid for uid, u in list(uploads.items())
            if (now - datetime.fromisoformat(u["created_at"])).total_seconds()
            > UPLOAD_TTL_SECONDS
        ]
        for uid in expired_uploads:
            upload = uploads.pop(uid, None)
            if upload:
                try:
                    Path(upload["path"]).unlink(missing_ok=True)
                except Exception:
                    pass
        if expired_uploads:
            logger.info("Cleanup: removed %d stale upload session(s)", len(expired_uploads))


@asynccontextmanager
async def lifespan(application: FastAPI):
    t = threading.Thread(target=_cleanup_loop, daemon=True, name="cleanup-loop")
    t.start()
    logger.info("Cleanup loop started (job TTL=%ds, upload TTL=%ds)", JOB_TTL_SECONDS, UPLOAD_TTL_SECONDS)
    yield
    _stop_cleanup.set()
    t.join(timeout=5)
    logger.info("Cleanup loop stopped")


app = FastAPI(title="Medical Summary Builder", lifespan=lifespan)


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

# ── in-memory stores ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}
uploads: dict[str, dict] = {}  # upload_id → {path, size}

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
        except Exception:
            pass


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── chunked upload endpoints ───────────────────────────────────────────────

@app.post("/api/upload/start")
async def upload_start():
    """Create a new upload session and return its ID."""
    upload_id = str(uuid.uuid4())
    uploads[upload_id] = {
        "path": WORK_DIR / f"{upload_id}_input.pdf",
        "size": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("[upload:%s] session started", upload_id[:8])
    return {"upload_id": upload_id}


@app.post("/api/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, request: Request):
    """Receive one binary chunk (application/octet-stream) and append to file."""
    upload = uploads.get(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload session not found")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty chunk body")

    path: Path = upload["path"]
    mode = "wb" if chunk_index == 0 else "ab"
    with open(path, mode) as fh:
        fh.write(body)
    upload["size"] += len(body)

    logger.info(
        "[upload:%s] chunk %d received (%d B, total %d B)",
        upload_id[:8], chunk_index, len(body), upload["size"],
    )
    return {"chunk": chunk_index, "received": len(body), "total_size": upload["size"]}


# ── pipeline trigger ───────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    upload_id: str
    model: str = "grok-4-fast"
    layout: Optional[str] = None


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    if not os.getenv("AI_BUILDER_TOKEN"):
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN not configured on server.")

    upload = uploads.pop(req.upload_id, None)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload session not found or already used.")

    pdf_path: Path = upload["path"]
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail="Uploaded file not found on server.")

    job_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"summary_{job_id}.docx"
    layout_clean: str | None = req.layout.strip() if req.layout and req.layout.strip() else None

    jobs[job_id] = {
        "status": "pending",
        "message": "Job queued…",
        "stage": -1,
        "output_path": None,
        "validation_issues": [],
        "completion_through": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "[job:%s] queued | upload=%s model=%s layout=%s size=%dB",
        job_id[:8], req.upload_id[:8], req.model, repr(layout_clean), upload["size"],
    )

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, pdf_path, DEFAULT_TEMPLATE, output_path, req.model, layout_clean),
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
