"""SHA256-based PDF extraction cache.

Cache files are stored in the project-root `cache/` directory.
Cache key = SHA256 of the PDF file bytes, ensuring the same file
at a different path still hits the cache, and a modified file always misses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import PageContent, PDFDocument

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[2] / "cache"


def _pdf_hash(pdf_path: Path) -> str:
    """Return the SHA256 hex digest of *pdf_path*."""
    h = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(pdf_hash: str, pdf_stem: str) -> Path:
    return CACHE_DIR / f"{pdf_hash[:8]}_{pdf_stem}.json"


def load_cache(pdf_path: Path) -> tuple[PDFDocument, str, float] | None:
    """Try to load a cached PDFDocument for *pdf_path*.

    Returns ``(PDFDocument, pdf_hash, quality_score)`` on cache hit, or ``None``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_hash = _pdf_hash(pdf_path)
    cache_file = _cache_path(pdf_hash, pdf_path.stem)

    if not cache_file.exists():
        logger.debug("Cache miss for %s (hash %s)", pdf_path.name, pdf_hash[:8])
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if data.get("pdf_hash") != pdf_hash:
            logger.warning("Cache hash mismatch — ignoring stale cache: %s", cache_file)
            return None

        pages = [PageContent(**p) for p in data["pages"]]
        doc = PDFDocument(path=pdf_path, pages=pages)
        quality = float(data.get("quality_score", 0.0))
        logger.info(
            "Cache hit for %s (%d pages, quality=%.2f)",
            pdf_path.name,
            doc.total_pages,
            quality,
        )
        return doc, pdf_hash, quality

    except Exception as exc:
        logger.warning("Failed to read cache %s: %s — will re-extract", cache_file, exc)
        return None


def save_cache(
    pdf_path: Path,
    pdf_hash: str,
    doc: PDFDocument,
    quality_score: float,
) -> None:
    """Persist *doc* to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(pdf_hash, pdf_path.stem)

    payload = {
        "pdf_hash": pdf_hash,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path),
        "total_pages": doc.total_pages,
        "quality_score": quality_score,
        "pages": [{"page_number": p.page_number, "text": p.text} for p in doc.pages],
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved cache → %s", cache_file)
