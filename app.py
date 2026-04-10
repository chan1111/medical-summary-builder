"""FastAPI web server for Medical Summary Builder."""

from __future__ import annotations

import json
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

from dotenv import load_dotenv
load_dotenv()

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

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEFAULT_TEMPLATE = BASE_DIR / "docs" / "summary_template.docx"
WORK_DIR = Path(tempfile.gettempdir()) / "medsummary"
OUTPUT_DIR = WORK_DIR / "output"
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── TTL constants ──────────────────────────────────────────────────────────
JOB_TTL_SECONDS = 2 * 60 * 60       # completed/errored jobs kept for 2 hours
UPLOAD_TTL_SECONDS = 15 * 60        # stale upload sessions expire after 15 minutes
CLEANUP_INTERVAL_SECONDS = 5 * 60   # run cleanup every 5 minutes

# ── file-based job store ───────────────────────────────────────────────────
# Using files instead of in-memory dicts so that multiple uvicorn worker
# processes can share state without conflict.

def _job_file(job_id: str) -> Path:
    return WORK_DIR / f"job_{job_id}.json"


def _upload_file(upload_id: str) -> Path:
    return WORK_DIR / f"upload_{upload_id}.json"


def _write_json(path: Path, data: dict) -> None:
    """Atomic write via temp file + rename (safe across processes)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# Job helpers
def _create_job(job_id: str, data: dict) -> None:
    _write_json(_job_file(job_id), data)


def _update_job(job_id: str, **kwargs) -> None:
    job = _read_json(_job_file(job_id)) or {}
    job.update(kwargs)
    _write_json(_job_file(job_id), job)


def _get_job(job_id: str) -> dict | None:
    return _read_json(_job_file(job_id))


# Upload helpers
def _create_upload(upload_id: str, data: dict) -> None:
    _write_json(_upload_file(upload_id), data)


def _get_upload(upload_id: str) -> dict | None:
    return _read_json(_upload_file(upload_id))


def _pop_upload(upload_id: str) -> dict | None:
    """Read upload metadata and delete the metadata file atomically."""
    data = _read_json(_upload_file(upload_id))
    _upload_file(upload_id).unlink(missing_ok=True)
    return data


def _update_upload_size(upload_id: str, added_bytes: int) -> int:
    """Increment stored size; returns new total. Not race-safe for parallel
    chunk uploads, but the frontend sends chunks sequentially so this is fine."""
    upload = _read_json(_upload_file(upload_id)) or {}
    upload["size"] = upload.get("size", 0) + added_bytes
    _write_json(_upload_file(upload_id), upload)
    return upload["size"]


# ── cleanup loop ───────────────────────────────────────────────────────────
_stop_cleanup = threading.Event()


def _cleanup_loop() -> None:
    """Background thread: evict expired job and upload metadata files."""
    while not _stop_cleanup.wait(timeout=CLEANUP_INTERVAL_SECONDS):
        now = datetime.now(timezone.utc)
        expired_jobs = 0
        expired_uploads = 0

        for p in list(WORK_DIR.glob("job_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("status") not in ("done", "error"):
                    continue
                created = datetime.fromisoformat(data["created_at"])
                if (now - created).total_seconds() > JOB_TTL_SECONDS:
                    if data.get("output_path"):
                        Path(data["output_path"]).unlink(missing_ok=True)
                    p.unlink(missing_ok=True)
                    expired_jobs += 1
            except Exception:
                pass

        for p in list(WORK_DIR.glob("upload_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(data["created_at"])
                if (now - created).total_seconds() > UPLOAD_TTL_SECONDS:
                    if data.get("path"):
                        Path(data["path"]).unlink(missing_ok=True)
                    p.unlink(missing_ok=True)
                    expired_uploads += 1
            except Exception:
                pass

        if expired_jobs:
            logger.info("Cleanup: removed %d expired job(s)", expired_jobs)
        if expired_uploads:
            logger.info("Cleanup: removed %d stale upload session(s)", expired_uploads)


@asynccontextmanager
async def lifespan(application: FastAPI):
    t = threading.Thread(target=_cleanup_loop, daemon=True, name="cleanup-loop")
    t.start()
    logger.info(
        "Cleanup loop started (job TTL=%ds, upload TTL=%ds)",
        JOB_TTL_SECONDS, UPLOAD_TTL_SECONDS,
    )
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
        _update_job(job_id, status="running", message=PIPELINE_STAGES[0], stage=0)

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
                    _update_job(job_id, message=msg, stage=i)
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
        _update_job(
            job_id,
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
        _update_job(job_id, status="error", message=str(exc))

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
    _create_upload(upload_id, {
        "path": str(WORK_DIR / f"{upload_id}_input.pdf"),
        "size": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("[upload:%s] session started", upload_id[:8])
    return {"upload_id": upload_id}


@app.post("/api/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, request: Request):
    """Receive one binary chunk (application/octet-stream) and append to file."""
    upload = _get_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload session not found")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty chunk body")

    path = Path(upload["path"])
    mode = "wb" if chunk_index == 0 else "ab"
    with open(path, mode) as fh:
        fh.write(body)

    new_size = _update_upload_size(upload_id, len(body))

    logger.info(
        "[upload:%s] chunk %d received (%d B, total %d B)",
        upload_id[:8], chunk_index, len(body), new_size,
    )
    return {"chunk": chunk_index, "received": len(body), "total_size": new_size}


# ── pipeline trigger ───────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    upload_id: str
    model: str = "grok-4-fast"
    layout: Optional[str] = None


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    if not os.getenv("AI_BUILDER_TOKEN"):
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN not configured on server.")

    upload = _pop_upload(req.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload session not found or already used.")

    pdf_path = Path(upload["path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail="Uploaded file not found on server.")

    job_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"summary_{job_id}.docx"
    layout_clean: str | None = req.layout.strip() if req.layout and req.layout.strip() else None

    _create_job(job_id, {
        "status": "pending",
        "message": "Job queued…",
        "stage": -1,
        "output_path": None,
        "validation_issues": [],
        "completion_through": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

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
    job = _get_job(job_id)
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
    job = _get_job(job_id)
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
