"""
app/core/variant_store.py
-------------------------
Manages saved resume variants: writes LaTeX + PDF + meta.json to a structured
local folder, indexes in the resume_variants DB table, and supports retrieval
by JD similarity (via ChromaDB jd_col collection + job_id lookup).

Folder layout:
    data/resumes/user_{id}/{variant_id}__{name_slug}/
        resume.tex
        resume.pdf
        meta.json
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.utils.sqlite_handler import SQLHandler
from app.utils.logger import get_logger
from config.config import get_config_dict

log = get_logger(__name__)
cfg = get_config_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _compile_latex(tex_path: Path) -> tuple[Path | None, int]:
    """
    Compile a .tex file with pdflatex (two passes).
    Returns (pdf_path, page_count). pdf_path is None on failure.
    Page count is parsed from pdflatex stdout; defaults to 1.
    """
    out_dir = tex_path.parent
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]
    page_count = 1
    try:
        for _ in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        # Parse "Output written on ... (N pages, M bytes)"
        match = re.search(r"Output written on .+? \((\d+) page", result.stdout)
        if match:
            page_count = int(match.group(1))
        pdf_path = tex_path.with_suffix(".pdf")
        return (pdf_path if pdf_path.exists() else None), page_count
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning("pdflatex unavailable or timed out: %s", exc)
        return None, page_count


# ──────────────────────────────────────────────────────────────────────────────
# VariantStore
# ──────────────────────────────────────────────────────────────────────────────

class VariantStore:
    def __init__(self) -> None:
        self.db = SQLHandler()
        data_dir = Path(cfg["path_dir"]["data_dir"])
        self.base_dir = data_dir / "resumes"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── internal ────────────────────────────────────────────────────────────

    def _user_dir(self, user_id: int) -> Path:
        d = self.base_dir / f"user_{user_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _enrich(row: dict) -> dict:
        """Parse JSON columns back into Python lists."""
        for field in ("tags", "job_ids", "session_ids"):
            raw = row.get(field)
            if isinstance(raw, str):
                try:
                    row[field] = json.loads(raw)
                except Exception:
                    row[field] = []
        return row

    # ── write ────────────────────────────────────────────────────────────────

    def save(
        self,
        user_id: int,
        name: str,
        latex: str,
        variant_type: str = "custom",
        base_resume_id: int | None = None,
        tags: list[str] | None = None,
        job_ids: list[int] | None = None,
        session_ids: list[int] | None = None,
    ) -> dict:
        """
        Write a variant to disk and record it in the DB.
        Returns the full variant dict including the assigned id.
        """
        slug = _slugify(name)
        user_dir = self._user_dir(user_id)

        # Insert DB row first to get the id, then rename folder to include it
        row_id = self.db.add_one(
            "resume_variants",
            {
                "user_id":        user_id,
                "name":           name,
                "variant_type":   variant_type,
                "base_resume_id": base_resume_id,
                "folder_path":    "__placeholder__",
                "page_count":     1,
                "tags":           json.dumps(tags or []),
                "job_ids":        json.dumps(job_ids or []),
                "session_ids":    json.dumps(session_ids or []),
            },
        )

        variant_dir = user_dir / f"{row_id}__{slug}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Write LaTeX
        tex_path = variant_dir / "resume.tex"
        tex_path.write_text(latex, encoding="utf-8")

        # Compile PDF
        _, page_count = _compile_latex(tex_path)

        # Write meta.json
        meta = {
            "id":             row_id,
            "user_id":        user_id,
            "name":           name,
            "variant_type":   variant_type,
            "base_resume_id": base_resume_id,
            "tags":           tags or [],
            "job_ids":        job_ids or [],
            "session_ids":    session_ids or [],
            "page_count":     page_count,
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }
        (variant_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Update DB with real folder path and page_count
        rel_path = str(variant_dir.relative_to(Path(cfg["path_dir"]["base_dir"])))
        self.db.update_data(
            "resume_variants",
            {"folder_path": rel_path, "page_count": page_count},
            {"id": row_id},
        )
        meta["folder_path"] = rel_path
        return meta

    # ── read ─────────────────────────────────────────────────────────────────

    def list(self, user_id: int) -> list[dict]:
        df = self.db.fetch_table_where("resume_variants", filters={"user_id": user_id})
        return [self._enrich(r) for r in df.to_dict("records")]

    def get(self, variant_id: int) -> dict | None:
        row = self.db.fetch_one("resume_variants", {"id": variant_id})
        return self._enrich(row) if row else None

    def get_latex(self, variant_id: int) -> str | None:
        row = self.db.fetch_one("resume_variants", {"id": variant_id})
        if not row:
            return None
        folder = Path(cfg["path_dir"]["base_dir"]) / row["folder_path"]
        tex = folder / "resume.tex"
        return tex.read_text(encoding="utf-8") if tex.exists() else None

    def get_pdf_path(self, variant_id: int) -> Path | None:
        row = self.db.fetch_one("resume_variants", {"id": variant_id})
        if not row:
            return None
        folder = Path(cfg["path_dir"]["base_dir"]) / row["folder_path"]
        pdf = folder / "resume.pdf"
        return pdf if pdf.exists() else None

    # ── delete ───────────────────────────────────────────────────────────────

    def delete(self, variant_id: int) -> bool:
        row = self.db.fetch_one("resume_variants", {"id": variant_id})
        if not row:
            return False
        folder = Path(cfg["path_dir"]["base_dir"]) / row["folder_path"]
        if folder.exists():
            shutil.rmtree(folder)
        self.db.execute_raw(
            "DELETE FROM resume_variants WHERE id = :id", {"id": variant_id}
        )
        return True

    # ── JD similarity search ─────────────────────────────────────────────────

    def search_by_jd(self, jd_text: str, user_id: int, top_k: int = 10) -> list[dict]:
        """
        Find variants whose associated job descriptions are most similar to
        the given JD text.

        Strategy:
         1. Embed jd_text and query the existing jd_col ChromaDB collection
            to find the closest stored job_ids.
         2. Find variants that include any of those job_ids.
         3. Rank by best similarity score.
        """
        try:
            from app.core.chroma_client import get_chroma_client
            from app.core.embeddings import get_ef
            from config.config import get_config_dict as _cfg

            _c = _cfg()
            client = get_chroma_client()
            ef = get_ef()
            jd_col = client.get_or_create_collection(
                name=_c["chromadb"]["collection1"],
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            results = jd_col.query(query_texts=[jd_text], n_results=min(top_k * 3, 30))
            meta_list = results.get("metadatas", [[]])[0]
            dist_list = results.get("distances", [[]])[0]

            # Build {job_id: best_similarity_score} (distance is cosine distance → similarity = 1 - d)
            job_scores: dict[int, float] = {}
            for meta, dist in zip(meta_list, dist_list):
                jid = meta.get("job_id")
                if jid is None:
                    continue
                score = round(1.0 - float(dist), 4)
                if int(jid) not in job_scores or score > job_scores[int(jid)]:
                    job_scores[int(jid)] = score

        except Exception as exc:
            log.warning("JD similarity search failed (falling back to all): %s", exc)
            job_scores = {}

        # Fetch all variants for user
        all_variants = self.list(user_id)
        if not job_scores:
            return all_variants[:top_k]

        # Score each variant by the best match among its job_ids
        scored: list[tuple[float, dict]] = []
        unscored: list[dict] = []
        for v in all_variants:
            best = max((job_scores[jid] for jid in v.get("job_ids", []) if jid in job_scores), default=None)
            if best is not None:
                scored.append((best, {**v, "similarity": best}))
            else:
                unscored.append({**v, "similarity": 0.0})

        scored.sort(key=lambda t: t[0], reverse=True)
        return [v for _, v in scored[:top_k]] + unscored[: max(0, top_k - len(scored))]
