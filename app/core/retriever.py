import json
import os
import chromadb
from chromadb import Collection as ChromaCollection
from chromadb.utils import embedding_functions
from typing import Any, List

from app.core.chroma_client import get_chroma_client

from app.models.jd import ParsedJD
from app.utils import SQLHandler, get_logger

log = get_logger(__name__)

from config.config import get_config_dict
config = get_config_dict()

chroma_path     = config['chromadb']['path']
jd_col          = config['chromadb']['collection1']
resume_col      = config['chromadb']['collection2']
EMBEDDING_MODEL = config.get('rag_config', {}).get('embedding_model', 'all-mpnet-base-v2')
HF_TOKEN        = config.get('huggingface', {}).get('token') or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN", "")

if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

FLAT_SECTIONS  = config['resume_defaults']['flat_sections']
ATOMIC_SECTIONS = config['resume_defaults']['atomic_sections']

TOP_K = config['retriever_config']['top_k']
FETCH_MULTIPLIER = config['retriever_config']['fetch_multiplier']
WEIGHT_REQUIRED_SKILLS = config['retriever_config']['weight_req_skills']   # text[0]      hard requirements
WEIGHT_RESPONSIBILITY  = config['retriever_config']['weight_resp']   # text[1:-1]   day-to-day work
WEIGHT_NICE_TO_HAVE    = config['retriever_config']['weight_nice_to_have']   # text[-1]     soft requirements

class ResumeRetriever:
    def __init__(self, top_k = TOP_K, fetch_mult = FETCH_MULTIPLIER, 
                weight_req_skills = WEIGHT_REQUIRED_SKILLS,
                weight_resp = WEIGHT_RESPONSIBILITY,
                weight_nice_to_have = WEIGHT_NICE_TO_HAVE):

        self._n_resp = 0
        self.top_k = top_k
        self.fetch_mult = fetch_mult
        self.weight_skill = weight_req_skills
        self.weight_resp = weight_resp
        self.weight_nth = weight_nice_to_have

        self.db = SQLHandler()
        self._chroma_client = get_chroma_client()
        # Use same EF as resume_parser / jd_parser so query embeddings are consistent
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            token=HF_TOKEN or None,
        )
        self.col_resume: ChromaCollection = self._chroma_client.get_or_create_collection(
            name              = resume_col,
            embedding_function= self._ef,
            metadata          = {"hnsw:space": "cosine"},
        )
        self.col_jd: ChromaCollection = self._chroma_client.get_or_create_collection(
            name              = jd_col,
            embedding_function= self._ef,
            metadata          = {"hnsw:space": "cosine"},
        )

    def _fetch_jd(self, job_id) -> ParsedJD|None:
        row = self.db.fetch_one(table_name='jobs', filters={"id": job_id})
        if not row:
            return None
        else:
            parsed_data = json.loads(row['jd_parsed'])
            return ParsedJD(**parsed_data)
        
    def _rerank(
        self,
        chroma_result:      dict,
        n_responsibilities: int,
        top_k:              int,
    ) -> list[tuple[float, str, dict]]:
        """
        Two-stage fusion:
        Stage 1 — CombMNZ within each group (required, resp, nice_to_have)
                    group_score = sum(similarities) * n_appearances
        Stage 2 — Weighted CombSUM across groups
                    fused_score = sum(WEIGHT[g] * group_score[g])

        CombMNZ within responsibilities rewards docs matching multiple
        responsibilities. Weighted CombSUM across groups ensures required_skills
        outweighs the entire responsibilities group regardless of its size.
        """
        idx_required   = 0
        idx_resp_start = 1
        idx_resp_end   = 1 + n_responsibilities

        # Stage 1 accumulators per group
        # sim_sum[group][cid]    = sum of similarities across queries in that group
        # appearances[group][cid]= number of query results the doc appeared in
        sim_sum:     dict[str, dict[str, float]] = {"required": {}, "resp": {}, "nice": {}}
        appearances: dict[str, dict[str, int]]   = {"required": {}, "resp": {}, "nice": {}}
        meta_store:  dict[str, dict]             = {}

        for query_idx, (id_row, dist_row, meta_row) in enumerate(
            zip(
                chroma_result["ids"],
                chroma_result["distances"],
                chroma_result["metadatas"],
            )
        ):
            if query_idx == idx_required:
                group = "required"
            elif idx_resp_start <= query_idx < idx_resp_end:
                group = "resp"
            else:
                group = "nice"

            for cid, dist, meta in zip(id_row, dist_row, meta_row):
                sim = 1.0 - dist
                sim_sum[group][cid]      = sim_sum[group].get(cid, 0.0) + sim
                appearances[group][cid]  = appearances[group].get(cid, 0) + 1
                if cid not in meta_store:
                    meta_store[cid] = meta

        # Stage 1 — CombMNZ score per group per doc
        # combmnz[group][cid] = sim_sum * appearances
        combmnz: dict[str, dict[str, float]] = {"required": {}, "resp": {}, "nice": {}}
        for group in ("required", "resp", "nice"):
            for cid in sim_sum[group]:
                combmnz[group][cid] = sim_sum[group][cid] * appearances[group][cid]

        # Stage 2 — weighted CombSUM across groups
        all_ids = set(combmnz["required"]) | set(combmnz["resp"]) | set(combmnz["nice"])
        fused: dict[str, float] = {}

        for cid in all_ids:
            fused[cid] = (
                self.weight_skill * combmnz["required"].get(cid, 0.0)
                + self.weight_resp  * combmnz["resp"].get(cid, 0.0)
                + self.weight_nth    * combmnz["nice"].get(cid, 0.0)
            )

        ranked = sorted(
            [(score, cid, meta_store[cid]) for cid, score in fused.items()],
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        log.info(
            "_rerank | n_resp=%d candidates=%d top_k=%d returned=%d",
            n_responsibilities, len(fused), top_k, len(ranked),
        )
        return ranked
    
    def _apply_rerank(
        self,
        raw:                dict | Any,
        n_responsibilities: int,
        top_k:              int,
    ) -> list[tuple[float, str, dict]]:
        """
        Validates raw ChromaDB result then calls _rerank.
        Returns empty list if ChromaDB returned nothing.

        Parameters
        ----------
        raw                : direct return value of col_resume.query()
        n_responsibilities : len(jd_obj.responsibilities)
        top_k              : final number of results wanted

        Returns
        -------
        list of (fused_score, chroma_id, metadata) sorted desc, length <= top_k

        Example
        -------
            raw    = self.col_resume.query(...)
            ranked = _apply_rerank(raw, n_responsibilities=5, top_k=TOP_K)
            for score, cid, meta in ranked:
                sql_id = meta.get("sql_id")
        """
        if not raw or not raw.get("ids") or not any(raw["ids"]):
            log.warning("_apply_rerank received empty ChromaDB result")
            return []
        return self._rerank(raw, n_responsibilities, top_k)
        
    # ------------------------------------------------------------------
    # Internal: build the ChromaDB `where` clause from section/item/resume filters
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        section_name: str,
        item_name:    str | None,
        resume_ids:   list[int] | None,
    ) -> dict:
        """
        Compose a ChromaDB `where` filter from the available constraints.
        All active clauses are ANDed together.
        """
        clauses: list[dict] = [{"section_type": {"$eq": section_name}}]

        if item_name:
            clauses.append({"item_name": {"$eq": item_name}})

        if resume_ids:
            if len(resume_ids) == 1:
                clauses.append({"resume_id": {"$eq": resume_ids[0]}})
            else:
                clauses.append({"resume_id": {"$in": resume_ids}})

        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    def retrieve_top_results(
        self,
        text:         list[str],
        section_name: str,
        item_name:    str | None = None,
        resume_ids:   list[int] | None = None,
    ):
        where = self._build_where(section_name, item_name, resume_ids)
        res = self.col_resume.query(
            query_texts=text,
            n_results=self.fetch_mult * self.top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        return self._apply_rerank(res, self._n_resp, TOP_K)

    def retrieve(self, text: list[str], resume_ids: list[int] | None = None):
        """
        Retrieve top-k sections for all flat + atomic section types.

        resume_ids — when supplied, ChromaDB queries are scoped to only those
        resume IDs so that only one user's content is considered.  The SQL
        item-name enumeration is also scoped to those resumes so that items
        from other users don't appear in the results.
        """
        results = []

        for flat_sec in FLAT_SECTIONS:
            results.append((
                flat_sec, None,
                self.retrieve_top_results(text, flat_sec, resume_ids=resume_ids),
            ))

        for atom_sec in ATOMIC_SECTIONS:
            if resume_ids:
                # Filter item_name enumeration to the user's resumes via a SQL JOIN
                placeholders = ",".join(str(rid) for rid in resume_ids)
                raw = self.db.execute_raw(
                    f"""
                    SELECT DISTINCT rsi.item_name
                    FROM   resume_section_items rsi
                    JOIN   resume_sections rs ON rsi.section_id = rs.id
                    WHERE  rsi.section_name = :sec
                      AND  rs.resume_id IN ({placeholders})
                    """,
                    {"sec": atom_sec},
                )
                item_list = [r["item_name"] for r in (raw or []) if r.get("item_name")]
            else:
                items_df  = self.db.fetch_table_where(
                    table_name="resume_section_items",
                    columns=["item_name"],
                    filters={"section_name": atom_sec},
                )
                item_list = items_df["item_name"].unique().tolist()

            for item in item_list:
                results.append((
                    atom_sec, item,
                    self.retrieve_top_results(text, atom_sec, item, resume_ids=resume_ids),
                ))

        return results

    def run(self, job_id, jd_obj: ParsedJD | None, resume_ids: list[int] | None = None):
        if jd_obj is None:
            jd_obj = self._fetch_jd(job_id)
        if not jd_obj:
            log.error("Job not found: job_id=%d", job_id)
            return []

        self._n_resp = len(jd_obj.responsibilities)
        text = (
            [",".join(jd_obj.required_skills)]
            + jd_obj.responsibilities
            + [",".join(jd_obj.nice_to_have_skills)]
        )
        return self.retrieve(text=text, resume_ids=resume_ids)
        
    def rank_and_filter(
        self,
        run_results: list[tuple[str, str | None, list[tuple[float, str, dict]]]],
    ) -> list[dict]:
        """
        Takes direct output of .run():
            [(section_name, item_name, [(score, chroma_id, meta), ...]), ...]

        For atomic sections:
            - Fetches actual text from DB using metadata (resume_id, section_name, item_name)
            - Aggregates k chunk scores per item using exponentially decaying
            weighted sum (normalised), then sorts items by aggregated score desc.

        For flat sections:
            - Fetches actual text from DB using metadata.
            - Passes through as-is, no reordering.

        Returns:
            [
                {
                    "section_name": str,
                    "item_name":    str | None,
                    "scores":       [float, ...],   # k chunk scores, desc order
                    "rows":         [dict, ...],     # k db rows, same order as scores
                },
                ...
            ]
        """
        def fetch_row(meta: dict, item_name: str | None) -> dict:
            resume_id = meta.get('resume_id')
            sec_name  = meta.get('section_type')
            if item_name:
                return self.db.fetch_one(
                    table_name='resume_section_items',
                    filters={"resume_id": resume_id, "section_name": sec_name, "item_name": item_name}
                ) or {}
            else:
                return self.db.fetch_one(
                    table_name='resume_sections',
                    filters={"resume_id": resume_id, "section_name": sec_name}
                ) or {}

        flat_output: list[dict] = []
        atom_grouped: dict[str, dict[str, list[tuple[float, dict, dict]]]] = {}

        for sec, item, chunks in run_results:
            if sec in FLAT_SECTIONS:
                flat_output.append({
                    "section_name": sec,
                    "item_name":    None,
                    "scores":       [sc for sc, _, _ in chunks],
                    "rows":         [fetch_row(meta, None) for _, _, meta in chunks],
                })
            elif sec in ATOMIC_SECTIONS:
                atom_grouped.setdefault(sec, {}).setdefault(item, [])
                for sc, _, meta in chunks:
                    row = fetch_row(meta, item)
                    atom_grouped[sec][item].append((sc, meta, row))

        atom_output: list[dict] = []
        for section_name, items in atom_grouped.items():
            aggregated: list[tuple[str, float, list[tuple[float, dict, dict]]]] = []

            for item_name, chunks in items.items():
                weight_sum = sum(0.7 ** i for i in range(len(chunks)))
                agg_score  = sum(sc * (0.7 ** i) for i, (sc, _, _) in enumerate(chunks))
                agg_score  = agg_score / weight_sum if weight_sum > 0 else 0.0
                aggregated.append((item_name, agg_score, chunks))

            aggregated.sort(key=lambda x: x[1], reverse=True)

            for item_name, _, chunks in aggregated:
                atom_output.append({
                    "section_name": section_name,
                    "item_name":    item_name,
                    "scores":       [sc for sc, _, _ in chunks],
                    "rows":         [row for _, _, row in chunks],
                })

        log.info(
            "rank_and_filter | flat=%d atomic_items=%d",
            len(flat_output),
            len(atom_output),
        )
        return flat_output + atom_output

    def run_retriever(self, job_id, jd_obj: ParsedJD | None, resume_ids: list[int] | None = None):
        return self.rank_and_filter(self.run(job_id, jd_obj, resume_ids=resume_ids))
