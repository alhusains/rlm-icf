"""
Azure AI Search extraction engine.

For each ICF template section:
  1. Short-circuit: standard text / skipped / adaptation-skipped (no LLM call).
  2. Build search queries from section metadata (reuses rag_query.expand_queries).
  3. Search the Azure AI Search index for relevant documents.
  4. Assemble retrieved documents into a prompt with the ICF extraction task.
  5. Single LLM call to Azure OpenAI for structured generation.
  6. Parse the JSON response (reuses parse_extraction_json from extract.py).

The Azure AI Search index must be pre-populated with the protocol document(s)
before running the pipeline.  Use Azure AI Studio or the Azure SDK to create
and populate the index.

Interface
---------
AzureSearchExtractionEngine implements the same extract_variable(protocol_text, variable)
signature as all other backends so the pipeline can swap backends with zero changes.
"""

from __future__ import annotations

from icf.azure_search_prompts import build_azure_search_messages
from icf.debug_logger import ICFDebugLogger
from icf.extract import parse_extraction_json
from icf.rag_query import expand_queries
from icf.types import Evidence, ExtractionResult, TemplateVariable


class AzureSearchExtractionEngine:
    """Extraction engine backed by Azure AI Search + Azure OpenAI.

    Parameters
    ----------
    search_endpoint : str
        Azure AI Search service endpoint (e.g. https://my-search.search.windows.net).
    search_key : str
        API key for the Azure AI Search service.
    search_index : str
        Name of the search index containing the protocol document(s).
    model_name : str
        Azure OpenAI deployment name for the generator LLM.
    backend : str
        LLM provider: 'openai' or 'azure_openai'.
    backend_kwargs : dict
        Forwarded to rlm.clients.get_client() (api_key, azure_endpoint, etc.).
    search_top_k : int
        Number of documents to retrieve per query from Azure AI Search.
    num_queries : int
        Number of search queries generated per ICF section.
    use_semantic : bool
        Whether to use semantic search (requires semantic configuration on the index).
    semantic_config : str | None
        Name of the semantic configuration on the index (required if use_semantic=True).
    max_retries : int
        Number of generation attempts on JSON parse failure.
    verbose : bool
        Print detailed retrieval/generation information.
    """

    def __init__(
        self,
        search_endpoint: str,
        search_key: str,
        search_index: str,
        model_name: str,
        backend: str = "azure_openai",
        backend_kwargs: dict | None = None,
        search_top_k: int = 10,
        num_queries: int = 3,
        use_semantic: bool = False,
        semantic_config: str | None = None,
        max_retries: int = 2,
        verbose: bool = False,
        debug_logger: ICFDebugLogger | None = None,
    ) -> None:
        self.search_endpoint = search_endpoint
        self.search_key = search_key
        self.search_index = search_index
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.search_top_k = search_top_k
        self.num_queries = num_queries
        self.use_semantic = use_semantic
        self.semantic_config = semantic_config
        self.max_retries = max_retries
        self.verbose = verbose
        self.debug_logger = debug_logger  # accepted for interface parity

        # Build the Azure AI Search client
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self._search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=search_index,
            credential=AzureKeyCredential(search_key),
        )

        # Build one LLM client; reuse across all sections.
        from rlm.clients import get_client

        kwargs = dict(self.backend_kwargs)
        kwargs["model_name"] = self.model_name
        self.client = get_client(self.backend, kwargs)

    # ------------------------------------------------------------------
    # Public interface — matches ExtractionEngine.extract_variable()
    # ------------------------------------------------------------------

    def extract_variable(
        self,
        protocol_text: str,  # kept for interface compatibility; not used
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single ICF section using Azure AI Search + LLM."""
        if variable.adaptation_skipped:
            return self._make_adaptation_skipped_result(variable)
        if variable.is_standard_text:
            return self._make_standard_result(variable)
        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self._make_skipped_result(variable)

        last_result: ExtractionResult | None = None
        for attempt in range(1, self.max_retries + 1):
            result = self._run_search_extraction(variable)
            if result.status != "ERROR":
                return result
            last_result = result
            if attempt < self.max_retries:
                print(
                    f"[AZURE_SEARCH] Section {variable.section_id}: attempt "
                    f"{attempt}/{self.max_retries} error ({result.error}). Retrying ..."
                )
            else:
                print(
                    f"[AZURE_SEARCH] Section {variable.section_id}: all "
                    f"{self.max_retries} attempts failed. Last error: {result.error}"
                )

        assert last_result is not None
        return last_result

    # ------------------------------------------------------------------
    # Search + Generation pipeline
    # ------------------------------------------------------------------

    def _run_search_extraction(self, variable: TemplateVariable) -> ExtractionResult:
        # Step 1 — Generate search queries from section metadata
        queries = expand_queries(variable, num_queries=self.num_queries)
        if self.verbose:
            print(
                f"[AZURE_SEARCH] [{variable.section_id}] Queries: "
                + " | ".join(f'"{q[:60]}"' for q in queries)
            )

        # Step 2 — Search Azure AI Search with all queries, merge results
        all_documents = self._search_multi_query(queries)
        if self.verbose:
            print(
                f"[AZURE_SEARCH] [{variable.section_id}] Retrieved "
                f"{len(all_documents)} unique documents."
            )

        if not all_documents:
            return ExtractionResult(
                section_id=variable.section_id,
                heading=variable.heading,
                sub_section=variable.sub_section,
                status="NOT_FOUND",
                answer="",
                filled_template="",
                evidence=[],
                confidence="LOW",
                notes="No documents returned from Azure AI Search.",
                raw_response="",
            )

        # Step 3 — Build prompt and call LLM
        messages = build_azure_search_messages(variable, all_documents)
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
            print(
                f"[AZURE_SEARCH] [{variable.section_id}] "
                f"Raw response ({len(raw)} chars)."
            )

        return self._parse_response(raw, variable)

    # ------------------------------------------------------------------
    # Azure AI Search retrieval
    # ------------------------------------------------------------------

    def _search_multi_query(self, queries: list[str]) -> list[dict]:
        """Run multiple queries against Azure AI Search and merge results."""
        seen_ids: set[str] = set()
        merged: list[dict] = []

        for query in queries:
            docs = self._search(query)
            for doc in docs:
                # Use a document key to deduplicate; fall back to content hash
                doc_id = (
                    doc.get("id")
                    or doc.get("chunk_id")
                    or doc.get("metadata_storage_path")
                    or str(hash(str(doc)))
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)

        # Limit total documents to avoid exceeding context window
        max_docs = self.search_top_k * 2
        return merged[:max_docs]

    def _search(self, query: str) -> list[dict]:
        """Execute a single search query against Azure AI Search."""
        try:
            kwargs: dict = {
                "search_text": query,
                "top": self.search_top_k,
            }
            if self.use_semantic and self.semantic_config:
                kwargs["query_type"] = "semantic"
                kwargs["semantic_configuration_name"] = self.semantic_config

            results = self._search_client.search(**kwargs)
            return [dict(r) for r in results]
        except Exception as e:
            print(f"[AZURE_SEARCH] Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Response parsing (reuses logic from ExtractionEngine)
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, variable: TemplateVariable
    ) -> ExtractionResult:
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
                notes="Failed to parse JSON from Azure AI Search LLM response.",
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
    def _make_adaptation_skipped_result(
        variable: TemplateVariable,
    ) -> ExtractionResult:
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
