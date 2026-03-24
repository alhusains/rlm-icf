"""
RAG (Retrieval-Augmented Generation) extraction engine.

For each ICF template section:
  1. Short-circuit: standard text / skipped / adaptation-skipped (no LLM call).
  2. Multi-query expansion: generate N search queries from section metadata.
  3. Hybrid retrieval: BM25 + dense → RRF → top-20 small chunks.
  4. Cross-encoder reranking: re-score candidates → top-8.
  5. Parent chunk expansion: swap small chunks for their richer parent chunks.
  6. Context assembly: deduplicate parents, sort by page, trim to token budget.
  7. Structured generation: single LLM call with retrieved context + CoT schema.
  8. Response parsing: reuse parse_extraction_json() from extract.py.

The ProtocolIndex is built ONCE before the extraction loop in pipeline._build_engine()
and shared across all sections — embeddings are not re-computed per section.

Interface
---------
RAGExtractionEngine implements the same extract_variable(protocol_text, variable)
signature as ExtractionEngine and NaiveExtractionEngine so the pipeline can swap
backends with zero changes downstream.
"""

from __future__ import annotations

from icf.debug_logger import ICFDebugLogger
from icf.extract import parse_extraction_json
from icf.rag_index import Chunk, ProtocolIndex, RAGConfig, _count_tokens
from icf.rag_prompts import build_rag_messages
from icf.rag_query import expand_queries
from icf.rag_rerank import CrossEncoderReranker, NoOpReranker
from icf.types import Evidence, ExtractionResult, TemplateVariable
from rlm.clients import get_client


class RAGExtractionEngine:
    """RAG-based extraction: retrieve relevant chunks then synthesise in one LLM call.

    Parameters
    ----------
    protocol_index : ProtocolIndex
        Pre-built index (call index.build() before passing here).
    reranker : CrossEncoderReranker | NoOpReranker
        Reranker instance from rag_rerank.get_reranker(config).
    model_name : str
        LLM model / deployment name for the generator.
    backend : str
        LLM provider: 'openai', 'azure_openai', or 'vllm'.
    backend_kwargs : dict
        Forwarded to rlm.clients.get_client() (api_key, azure_endpoint, etc.).
    config : RAGConfig
        RAG configuration (retrieval_top_k, rerank_top_k, etc.).
    max_retries : int
        Number of generation attempts on JSON parse failure.
    """

    def __init__(
        self,
        protocol_index: ProtocolIndex,
        reranker: CrossEncoderReranker | NoOpReranker,
        model_name: str,
        backend: str,
        backend_kwargs: dict | None = None,
        config: RAGConfig | None = None,
        max_retries: int = 2,
        verbose: bool = False,
        debug_logger: ICFDebugLogger | None = None,
    ) -> None:
        self.index = protocol_index
        self.reranker = reranker
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.config = config or RAGConfig()
        self.max_retries = max_retries
        self.verbose = verbose
        self.debug_logger = debug_logger  # accepted for interface parity; not used

        # Build one LLM client; reuse across all sections.
        kwargs = dict(self.backend_kwargs)
        kwargs["model_name"] = self.model_name
        self.client = get_client(self.backend, kwargs)

    # ------------------------------------------------------------------
    # Public interface — matches ExtractionEngine.extract_variable()
    # ------------------------------------------------------------------

    def extract_variable(
        self,
        protocol_text: str,  # kept for interface compatibility; not used (index is pre-built)
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single ICF section using the RAG pipeline."""
        if variable.adaptation_skipped:
            return self._make_adaptation_skipped_result(variable)
        if variable.is_standard_text:
            return self._make_standard_result(variable)
        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self._make_skipped_result(variable)

        last_result: ExtractionResult | None = None
        for attempt in range(1, self.max_retries + 1):
            result = self._run_rag_extraction(variable)
            if result.status != "ERROR":
                return result
            last_result = result
            if attempt < self.max_retries:
                print(
                    f"[RAG] Section {variable.section_id}: attempt {attempt}/{self.max_retries} "
                    f"error ({result.error}). Retrying ..."
                )
            else:
                print(
                    f"[RAG] Section {variable.section_id}: all {self.max_retries} attempts "
                    f"failed. Last error: {result.error}"
                )

        assert last_result is not None
        return last_result

    # ------------------------------------------------------------------
    # RAG pipeline
    # ------------------------------------------------------------------

    def _run_rag_extraction(self, variable: TemplateVariable) -> ExtractionResult:
        # Step 1 — Multi-query expansion
        queries = expand_queries(variable, num_queries=self.config.num_queries)
        if self.verbose:
            print(
                f"[RAG] [{variable.section_id}] Queries: "
                + " | ".join(f'"{q[:60]}"' for q in queries)
            )

        # Step 2 — Hybrid retrieval (multi-query RRF)
        small_chunks = self.index.retrieve(queries, top_k=self.config.retrieval_top_k)
        if self.verbose:
            print(
                f"[RAG] [{variable.section_id}] Retrieved {len(small_chunks)} small chunks."
            )

        # Step 3 — Cross-encoder reranking (uses the primary query for scoring)
        primary_query = queries[0]
        reranked = self.reranker.rerank(primary_query, small_chunks, self.config.rerank_top_k)
        if self.verbose:
            print(
                f"[RAG] [{variable.section_id}] After reranking: {len(reranked)} chunks."
            )

        # Step 4 — Expand small → parent chunks + context assembly
        parent_chunks = self._expand_and_assemble(reranked)
        if self.verbose:
            total_tokens = sum(c.tokens for c in parent_chunks)
            print(
                f"[RAG] [{variable.section_id}] Context: {len(parent_chunks)} parent chunks, "
                f"~{total_tokens} tokens."
            )

        # Step 5 — Single structured LLM call
        messages = build_rag_messages(variable, parent_chunks)
        try:
            raw = self.client.completion(messages)
        except Exception as e:
            return ExtractionResult(
                section_id=variable.section_id,
                heading=variable.heading,
                sub_section=variable.sub_section,
                status="ERROR",
                answer="",
                filled_template="",
                evidence=[],
                confidence="LOW",
                notes="",
                raw_response="",
                error=f"{type(e).__name__}: {e}",
            )

        if self.verbose:
            print(f"[RAG] [{variable.section_id}] Raw response ({len(raw)} chars).")

        return self._parse_response(raw, variable)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _expand_and_assemble(self, small_chunks: list[Chunk]) -> list[Chunk]:
        """Expand small chunks → parent chunks, dedup, sort, trim to budget."""
        # Expand to parents; deduplicate by parent_id
        parent_map: dict[str, Chunk] = {}
        for small in small_chunks:
            parent = self.index.get_parent(small)
            parent_map[parent.chunk_id] = parent

        # Sort by document order (page_start)
        sorted_parents = sorted(parent_map.values(), key=lambda c: c.page_start)

        # Trim to context token budget
        budget = self.config.context_budget_tokens
        assembled: list[Chunk] = []
        tokens_used = 0
        for parent in sorted_parents:
            chunk_tokens = parent.tokens if parent.tokens > 0 else _count_tokens(parent.text)
            if tokens_used + chunk_tokens > budget:
                break
            assembled.append(parent)
            tokens_used += chunk_tokens

        return assembled

    # ------------------------------------------------------------------
    # Response parsing (reuses logic from ExtractionEngine)
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, variable: TemplateVariable) -> ExtractionResult:
        data = parse_extraction_json(raw)

        if data is None:
            return ExtractionResult(
                section_id=variable.section_id,
                heading=variable.heading,
                sub_section=variable.sub_section,
                status="ERROR",
                answer="",
                filled_template="",
                evidence=[],
                confidence="LOW",
                notes="Failed to parse JSON from RAG LLM response.",
                raw_response=raw,
                error="JSON parse failure",
            )

        evidence: list[Evidence] = []
        for e in data.get("evidence", []):
            if isinstance(e, dict):
                evidence.append(
                    Evidence(
                        quote=str(e.get("quote", "")),
                        page=str(e.get("page", "")),
                        section=str(e.get("section", "")),
                    )
                )

        return ExtractionResult(
            section_id=data.get("section_id", variable.section_id),
            heading=variable.heading,
            sub_section=variable.sub_section,
            status=data.get("status", "ERROR"),
            answer=data.get("answer", ""),
            filled_template=data.get("filled_template", ""),
            evidence=evidence,
            confidence=data.get("confidence", "LOW"),
            notes=data.get("notes", ""),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Short-circuit helpers (identical to the other backends)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_standard_result(variable: TemplateVariable) -> ExtractionResult:
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="STANDARD_TEXT",
            answer=variable.required_text,
            filled_template=variable.required_text,
            evidence=[],
            confidence="HIGH",
            notes="Standard required text — no extraction needed.",
        )

    @staticmethod
    def _make_adaptation_skipped_result(variable: TemplateVariable) -> ExtractionResult:
        reason = (
            variable.adaptation_notes
            or "Marked as not applicable for this study by adaptation pass."
        )
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="ADAPTATION_SKIPPED",
            answer="",
            filled_template="",
            evidence=[],
            confidence="N/A",
            notes=reason,
        )

    @staticmethod
    def _make_skipped_result(variable: TemplateVariable) -> ExtractionResult:
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="SKIPPED",
            answer="",
            filled_template="",
            evidence=[],
            confidence="N/A",
            notes=(
                "Section marked as 'Not in protocol — requires manual entry'. "
                "Use suggested text from template as a starting point."
            ),
        )
