"""
app/api/routes/library.py
--------------------------
Resume variant library endpoints.

  GET  /library/variants?user_id=1          — list all variants
  GET  /library/variants/{id}               — get single variant metadata
  POST /library/variants                    — save a new variant
  DELETE /library/variants/{id}             — delete a variant
  GET  /library/search?jd_text=...&user_id= — search by JD similarity
  GET  /library/variants/{id}/download      — download .tex or .pdf
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.variant_store import VariantStore

router = APIRouter(prefix="/library", tags=["Library"])


# ── request / response models ─────────────────────────────────────────────────

class SaveVariantRequest(BaseModel):
    user_id:        int
    name:           str
    latex:          str
    variant_type:   str = "custom"
    base_resume_id: int | None = None
    tags:           list[str] = []
    job_ids:        list[int] = []
    session_ids:    list[int] = []


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/variants")
def list_variants(user_id: int = Query(..., description="User ID")):
    store = VariantStore()
    return store.list(user_id)


@router.get("/variants/{variant_id}")
def get_variant(variant_id: int):
    store = VariantStore()
    v = store.get(variant_id)
    if not v:
        raise HTTPException(status_code=404, detail="Variant not found")
    return v


@router.post("/variants", status_code=201)
def save_variant(req: SaveVariantRequest):
    store = VariantStore()
    return store.save(
        user_id=req.user_id,
        name=req.name,
        latex=req.latex,
        variant_type=req.variant_type,
        base_resume_id=req.base_resume_id,
        tags=req.tags,
        job_ids=req.job_ids,
        session_ids=req.session_ids,
    )


@router.delete("/variants/{variant_id}", status_code=204)
def delete_variant(variant_id: int):
    store = VariantStore()
    if not store.delete(variant_id):
        raise HTTPException(status_code=404, detail="Variant not found")


@router.get("/search")
def search_by_jd(
    jd_text: str = Query(..., description="Job description text to match against"),
    user_id: int = Query(..., description="User ID"),
    top_k:   int = Query(10, ge=1, le=50),
):
    store = VariantStore()
    return store.search_by_jd(jd_text=jd_text, user_id=user_id, top_k=top_k)


@router.get("/variants/{variant_id}/download")
def download_variant(
    variant_id: int,
    format: str = Query("pdf", pattern="^(tex|pdf)$"),
):
    store = VariantStore()
    if format == "tex":
        latex = store.get_latex(variant_id)
        if not latex:
            raise HTTPException(status_code=404, detail="LaTeX file not found")
        from fastapi.responses import PlainTextResponse
        v = store.get(variant_id)
        name = v["name"] if v else f"variant_{variant_id}"
        return PlainTextResponse(
            latex,
            headers={"Content-Disposition": f'attachment; filename="{name}.tex"'},
        )
    else:
        pdf_path = store.get_pdf_path(variant_id)
        if not pdf_path:
            raise HTTPException(status_code=404, detail="PDF not found — was pdflatex available when this variant was saved?")
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.parent.name + ".pdf",
        )
