import json
import chromadb
from chromadb import Collection as ChromaCollection
from chromadb.config import Settings
from typing import Any, List

from data_models.jd_parse_dm import ParsedJD
from src.resume_parser import ATOMIC_SECTIONS
from utils.sqlite_handler import SQLHandler
from utils.app_logger import get_logger
log = get_logger(__name__)

from config.config import get_config_dict
config = get_config_dict()


chroma_path = config['chromadb']['path']
jd_col = config['chromadb']['collection1']
resume_col = config['chromadb']['collection2']

FLAT_SECTIONS = ['education', 'achievements', 'technical_skills', 'relevant_coursework']
ATOMIC_SECTIONS = ['experience', 'projects']
TOP_K = 10
FETCH_MULTIPLIER = 3
WEIGHT_REQUIRED_SKILLS = 1.0   # text[0]      hard requirements
WEIGHT_RESPONSIBILITY  = 0.8   # text[1:-1]   day-to-day work
WEIGHT_NICE_TO_HAVE    = 0.5   # text[-1]     soft requirements

class ResumeRetriever:
    def __init__(self):

        self._n_resp = 0

        self.db = SQLHandler()
        self._chroma_client = chromadb.PersistentClient(
            path     = chroma_path,
            settings = Settings(anonymized_telemetry=False),
        )
        self.col_resume: ChromaCollection = self._chroma_client.get_or_create_collection(
            name     = resume_col,
            metadata = {"hnsw:space": "cosine"},   # cosine distance throughout
        )
        self.col_jd: ChromaCollection = self._chroma_client.get_or_create_collection(
            name     = jd_col,
            metadata = {"hnsw:space": "cosine"},   # cosine distance throughout
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
                WEIGHT_REQUIRED_SKILLS * combmnz["required"].get(cid, 0.0)
                + WEIGHT_RESPONSIBILITY  * combmnz["resp"].get(cid, 0.0)
                + WEIGHT_NICE_TO_HAVE    * combmnz["nice"].get(cid, 0.0)
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
        
    def retrieve_top_results(self, text:list[str], section_name: str, item_name: str|None = None):
        res = []
        if item_name:
            res = self.col_resume.query(query_texts=text, 
                                        n_results=FETCH_MULTIPLIER*TOP_K, 
                                        where = {"section_type": section_name, "item_name": item_name},
                                        include = ["metadatas", "distances"])
        else:
            res = self.col_resume.query(query_texts=text, 
                                        n_results=FETCH_MULTIPLIER*TOP_K, 
                                        where = {"section_type": section_name},
                                        include = ["metadatas", "distances"])
            
        return self._apply_rerank(res if hasattr(res, 'to_dict') else res,  self._n_resp, TOP_K)

    def retrieve(self, text: list[str]):
        results = []
        for flat_sec in FLAT_SECTIONS:
            results.append((flat_sec, None, self.retrieve_top_results(text=text, section_name=flat_sec, item_name=None)))
        
        for atom_sec in ATOMIC_SECTIONS:
            items_df = (self.db.fetch_table_where(table_name='resume_section_items', columns=["item_name"], filters={"section_name": atom_sec}))
            item_list = items_df['item_name'].unique().tolist()
            for item in item_list:
                results.append((atom_sec, item, self.retrieve_top_results(text=text, section_name='atom_sec', item_name = item)))
        return results

    def run(self, job_id):
        jd_obj = self._fetch_jd(job_id)
        if not jd_obj:
            log.error("Job not found: job_id=%d", job_id)
            return []

        self._n_resp = len(jd_obj.responsibilities)
        text = [",".join(jd_obj.required_skills)]+ jd_obj.responsibilities+[",".join(jd_obj.nice_to_have_skills)]
        return self.retrieve(text=text)
        

        
        