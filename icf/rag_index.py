"""
Protocol indexing for the RAG extraction backend.

Builds a hybrid search index (BM25 + dense embeddings) over parent-child
chunks of the protocol.  The index is built once per pipeline run and shared
across all ICF section extractions.

Architecture
------------
DocumentParser
  → detects section headers and table-like regions in the raw protocol text
  → creates *parent chunks* (~800 tokens) as the generation context unit
  → splits each parent into overlapping *small chunks* (~200 tokens) as the
    retrieval unit (parent-child / "small-to-big" chunking)
  → table regions are kept as atomic parent+small chunk pairs to avoid
    destroying the column/row relationships that text splitters would break

BM25Index
  → keyword search over small chunks via BM25Okapi
  → excellent for exact clinical term matching (drug names, dosages, IDs)

DenseIndex
  → semantic search via OpenAI embeddings (text-embedding-3-large recommended)
  → supports Azure OpenAI and standard OpenAI backends
  → uses Matryoshka dimension reduction (3072 → 1024) for efficiency

HybridRetriever
  → fuses BM25 and dense rankings via Reciprocal Rank Fusion (RRF, k=60)
  → empirically the best retrieval fusion strategy with no hyperparameter tuning

ProtocolIndex
  → top-level orchestrator: build once, call retrieve() many times
  → multi-query retrieval: merges results from N search queries via a second
    RRF pass before handing off to the reranker
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from icf.types import IndexedProtocol

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """All tunable parameters for the RAG extraction backend."""

    # --- Embedding ---
    # Deployment name for Azure OpenAI or model name for standard OpenAI.
    # text-embedding-3-large is strongly recommended for clinical text.
    embedding_model: str = "text-embedding-3-large"
    # Reduced from the native 3072 dims via Matryoshka representation.
    # 1024 gives ~99% of full quality at 1/3 the memory cost.
    # Only applied to text-embedding-3-* models (ignored for ada-002).
    embedding_dimensions: int = 1024
    # Max texts per embeddings API call (Azure/OpenAI limit: 2048).
    embedding_batch_size: int = 100

    # --- Chunking ---
    # Small chunks are the retrieval unit; parent chunks are the generation unit.
    small_chunk_tokens: int = 200
    parent_chunk_tokens: int = 800
    # Overlap between adjacent small chunks within the same parent.
    chunk_overlap_tokens: int = 50

    # --- Retrieval ---
    retrieval_top_k: int = 20  # candidates before reranking
    rerank_top_k: int = 8  # final chunks passed to the generator LLM
    num_queries: int = 4  # search queries generated per ICF section

    # --- Reranker ---
    # "local"  → cross-encoder/ms-marco-MiniLM-L-12-v2 (sentence-transformers)
    # "none"   → skip reranking (ablation / faster runs)
    reranker: str = "local"

    # --- Context assembly ---
    # Token budget for the retrieved context block passed to the generator.
    # Generous budget; GPT-4o / GPT-5 have 128k context so this is fine.
    context_budget_tokens: int = 6000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval and context assembly."""

    chunk_id: str
    text: str
    tokens: int
    page_start: int
    page_end: int
    section_header: str  # most recent section heading seen before this chunk
    is_table: bool
    # None for parent chunks; set to parent.chunk_id for small (retrieval) chunks.
    parent_id: str | None = None


@dataclass
class _Block:
    """Intermediate unit: a single paragraph or table region from the protocol."""

    text: str
    page: int
    is_table: bool
    section_header: str


# ---------------------------------------------------------------------------
# Regex patterns for structure detection
# ---------------------------------------------------------------------------

_PAGE_RE = re.compile(r"--- PAGE (\d+) ---")

# Numbered section: "7.2 Safety Monitoring" / "1. Introduction"
_NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")

# Pure uppercase heading: "SAFETY MONITORING" (2-10 words)
_UPPER_SECTION_RE = re.compile(r"^[A-Z][A-Z\s\-/]{6,70}$")

# Explicit keyword: "SECTION X", "APPENDIX A", "SCHEDULE OF ACTIVITIES"
_KEYWORD_SECTION_RE = re.compile(
    r"^(?:SECTION|APPENDIX|SCHEDULE|ATTACHMENT|ADDENDUM)\s+\S",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------


class DocumentParser:
    """Converts raw protocol text into parent + small chunks.

    The input is the IndexedProtocol.full_text string which already has
    '--- PAGE X ---' markers from the ingest stage.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

    def parse(self, full_text: str) -> tuple[list[Chunk], dict[str, Chunk]]:
        """Parse protocol text into retrieval-ready chunks.

        Returns:
            small_chunks: flat list of Chunk objects indexed by BM25 + Dense.
            parent_map:   {parent_id → parent Chunk} for context expansion.
        """
        blocks = self._extract_blocks(full_text)
        parents = self._group_into_parents(blocks)

        small_chunks: list[Chunk] = []
        for parent in parents:
            small_chunks.extend(self._split_into_small(parent))

        parent_map = {p.chunk_id: p for p in parents}
        return small_chunks, parent_map

    # ------------------------------------------------------------------
    # Block extraction
    # ------------------------------------------------------------------

    def _extract_blocks(self, full_text: str) -> list[_Block]:
        """Split full_text on page markers and blank lines → list of _Block."""
        blocks: list[_Block] = []
        current_page = 1
        current_section = ""

        # _PAGE_RE.split gives: [pre, page_num, content, page_num, content, ...]
        parts = _PAGE_RE.split(full_text)

        i = 0
        while i < len(parts):
            part = parts[i]
            if part.strip().isdigit():
                current_page = int(part.strip())
                i += 1
                continue

            for para in re.split(r"\n{2,}", part):
                para = para.strip()
                if not para:
                    continue

                first_line = para.split("\n")[0].strip()
                if self._is_section_header(first_line):
                    current_section = first_line

                blocks.append(
                    _Block(
                        text=para,
                        page=current_page,
                        is_table=self._is_table(para),
                        section_header=current_section,
                    )
                )
            i += 1

        return blocks

    @staticmethod
    def _is_section_header(line: str) -> bool:
        if not line or len(line) > 120:
            return False
        return bool(
            _NUMBERED_SECTION_RE.match(line)
            or (
                _UPPER_SECTION_RE.match(line)
                and 2 <= len(line.split()) <= 12
                and not line.endswith(".")
            )
            or _KEYWORD_SECTION_RE.match(line)
        )

    @staticmethod
    def _is_table(text: str) -> bool:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) < 3:
            return False

        # Pipe-delimited table (most reliable signal from python-docx output)
        pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
        if pipe_lines / len(lines) >= 0.4:
            return True

        # Column-aligned table: many very short tokens per line
        # (e.g. "X" marks in a Schedule of Activities)
        def _short_ratio(ln: str) -> float:
            toks = ln.split()
            if not toks:
                return 0.0
            return sum(1 for t in toks if len(t) <= 2) / len(toks)

        avg_short = sum(_short_ratio(ln) for ln in lines) / len(lines)
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        return avg_short > 0.45 and avg_len > 25

    # ------------------------------------------------------------------
    # Parent grouping
    # ------------------------------------------------------------------

    def _group_into_parents(self, blocks: list[_Block]) -> list[Chunk]:
        """Group consecutive blocks into parent chunks (~parent_chunk_tokens)."""
        parents: list[Chunk] = []
        buf_blocks: list[_Block] = []
        buf_tokens = 0
        parent_idx = 0

        def _flush(page_end: int) -> None:
            nonlocal parent_idx, buf_blocks, buf_tokens
            if not buf_blocks:
                return
            text = "\n\n".join(b.text for b in buf_blocks)
            parents.append(
                Chunk(
                    chunk_id=f"p{parent_idx}",
                    text=text,
                    tokens=buf_tokens,
                    page_start=buf_blocks[0].page,
                    page_end=page_end,
                    section_header=buf_blocks[-1].section_header,
                    is_table=False,
                    parent_id=None,
                )
            )
            parent_idx += 1
            buf_blocks.clear()
            buf_tokens = 0

        for block in blocks:
            block_tokens = _count_tokens(block.text)

            # Tables → standalone atomic parent (never merged)
            if block.is_table:
                _flush(block.page)
                parents.append(
                    Chunk(
                        chunk_id=f"p{parent_idx}",
                        text=block.text,
                        tokens=block_tokens,
                        page_start=block.page,
                        page_end=block.page,
                        section_header=block.section_header,
                        is_table=True,
                        parent_id=None,
                    )
                )
                parent_idx += 1
                continue

            # New section header → flush current buffer to keep sections clean
            if block.section_header and buf_blocks and block.section_header != buf_blocks[-1].section_header:
                _flush(block.page)

            # Overflow → flush
            if buf_tokens + block_tokens > self.config.parent_chunk_tokens and buf_blocks:
                _flush(block.page)

            buf_blocks.append(block)
            buf_tokens += block_tokens

        _flush(blocks[-1].page if blocks else 1)
        return parents

    # ------------------------------------------------------------------
    # Small chunk splitting
    # ------------------------------------------------------------------

    def _split_into_small(self, parent: Chunk) -> list[Chunk]:
        """Split parent into overlapping small chunks for retrieval.

        Tables are not split — they become a single small chunk equal to
        the parent, ensuring the full table goes to the generator.
        Small parents (≤ small_chunk_tokens) also become a single chunk.
        """
        if parent.is_table or parent.tokens <= self.config.small_chunk_tokens:
            return [
                Chunk(
                    chunk_id=f"{parent.chunk_id}_s0",
                    text=parent.text,
                    tokens=parent.tokens,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    section_header=parent.section_header,
                    is_table=parent.is_table,
                    parent_id=parent.chunk_id,
                )
            ]

        if _ENC is None:
            # tiktoken not installed: fall back to whitespace-token approximation
            return self._split_by_words(parent)

        all_tokens = _ENC.encode(parent.text)
        size = self.config.small_chunk_tokens
        overlap = self.config.chunk_overlap_tokens
        step = max(size - overlap, 1)

        small: list[Chunk] = []
        idx = 0
        while idx < len(all_tokens):
            end = min(idx + size, len(all_tokens))
            chunk_toks = all_tokens[idx:end]
            chunk_text = _ENC.decode(chunk_toks)
            small.append(
                Chunk(
                    chunk_id=f"{parent.chunk_id}_s{len(small)}",
                    text=chunk_text,
                    tokens=len(chunk_toks),
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    section_header=parent.section_header,
                    is_table=False,
                    parent_id=parent.chunk_id,
                )
            )
            if end == len(all_tokens):
                break
            idx += step

        return small

    def _split_by_words(self, parent: Chunk) -> list[Chunk]:
        """Word-based fallback splitter when tiktoken is unavailable."""
        words = parent.text.split()
        size = self.config.small_chunk_tokens  # approximate: 1 word ≈ 1 token
        overlap = self.config.chunk_overlap_tokens
        step = max(size - overlap, 1)

        small: list[Chunk] = []
        idx = 0
        while idx < len(words):
            chunk_words = words[idx : idx + size]
            chunk_text = " ".join(chunk_words)
            small.append(
                Chunk(
                    chunk_id=f"{parent.chunk_id}_s{len(small)}",
                    text=chunk_text,
                    tokens=len(chunk_words),
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    section_header=parent.section_header,
                    is_table=False,
                    parent_id=parent.chunk_id,
                )
            )
            if idx + size >= len(words):
                break
            idx += step

        return small


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text.split())  # word-count approximation


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """Keyword search over small chunks using BM25Okapi."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_bm25_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs by BM25 relevance."""
        if self._bm25 is None or not self._chunks:
            return []
        scores: np.ndarray = self._bm25.get_scores(_bm25_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Dense (Embedding) Index
# ---------------------------------------------------------------------------


class DenseIndex:
    """Semantic search via OpenAI embeddings + cosine similarity (numpy)."""

    def __init__(self, config: RAGConfig, embedding_client: Any) -> None:
        self.config = config
        self._client = embedding_client
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n_chunks, dims)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._matrix = self._embed_texts([c.text for c in chunks])

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, cosine_similarity) pairs."""
        if self._matrix is None or not self._chunks:
            return []
        q_vec = self._embed_texts([query])[0]
        sims = _cosine_sim(q_vec, self._matrix)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[i], float(sims[i])) for i in top_idx]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        batch_size = self.config.embedding_batch_size
        vecs: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs: dict[str, Any] = {
                "model": self.config.embedding_model,
                "input": batch,
            }
            # Matryoshka dimension reduction (text-embedding-3-* only)
            if "text-embedding-3" in self.config.embedding_model:
                kwargs["dimensions"] = self.config.embedding_dimensions
            response = self._client.embeddings.create(**kwargs)
            vecs.extend(e.embedding for e in response.data)

        return np.array(vecs, dtype=np.float32)


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return (m_norms @ q_norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Hybrid Retriever (RRF fusion)
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Fuses BM25 and dense rankings via Reciprocal Rank Fusion.

    RRF score = Σ  1 / (k + rank_i)  for each system i.
    k=60 is the standard value (Cormack et al., 2009).
    """

    RRF_K: int = 60

    def __init__(self, bm25: BM25Index, dense: DenseIndex, config: RAGConfig) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.config = config

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Single-query hybrid retrieval returning top_k chunks."""
        n_candidates = min(top_k * 3, top_k + 60)

        bm25_results = self.bm25.search(query, n_candidates)
        dense_results = self.dense.search(query, n_candidates)

        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                self.RRF_K + rank + 1
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                self.RRF_K + rank + 1
            )
            chunk_map[chunk.chunk_id] = chunk

        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [chunk_map[cid] for cid in ranked]


# ---------------------------------------------------------------------------
# Embedding client factory
# ---------------------------------------------------------------------------


def build_embedding_client(backend: str, backend_kwargs: dict) -> Any:
    """Create an OpenAI-compatible client for the embeddings endpoint.

    Supports the same backend names as the rest of the pipeline:
      "openai"        → openai.OpenAI
      "azure_openai"  → openai.AzureOpenAI
      "vllm"          → openai.OpenAI with custom base_url

    Azure — separate resource support
    ----------------------------------
    The embedding model may be deployed in a different Azure region / resource
    than the LLM.  This function therefore reads *embedding-specific* env vars
    first and only falls back to the shared LLM vars when they are absent:

      AZURE_OPENAI_EMBEDDING_API_KEY      (falls back to AZURE_OPENAI_API_KEY)
      AZURE_OPENAI_EMBEDDING_ENDPOINT     (falls back to AZURE_OPENAI_ENDPOINT)
      AZURE_OPENAI_EMBEDDING_API_VERSION  (falls back to AZURE_OPENAI_API_VERSION,
                                           then "2024-02-01")

    All four variables can be set in your .env file.
    """
    if backend == "azure_openai":
        from openai import AzureOpenAI

        api_key = (
            os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY")
            or backend_kwargs.get("api_key")
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        azure_endpoint = (
            os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            or backend_kwargs.get("azure_endpoint")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        # 2024-02-01 is the minimum version that supports the 'dimensions'
        # parameter for text-embedding-3-* models.
        api_version = (
            os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION")
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or "2024-02-01"
        )

        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
    else:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if api_key := backend_kwargs.get("api_key"):
            kwargs["api_key"] = api_key
        if base_url := backend_kwargs.get("base_url"):
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------


def _extract_protocol_id(source_path: str) -> str:
    """Extract the leading numeric ID from a protocol filename.

    '21-5995_REBApprovedProtocol.docx' → '21-5995'
    Falls back to the full stem (first 48 chars) for non-standard names.
    """
    name = os.path.splitext(os.path.basename(source_path))[0]
    m = re.match(r"^(\d[\d\-]*)", name)
    return m.group(1) if m else name[:48]


def _text_fingerprint(text: str) -> str:
    """8-char MD5 hex of the protocol text — detects file changes cheaply."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# ProtocolIndex — top-level orchestrator
# ---------------------------------------------------------------------------


class ProtocolIndex:
    """Pre-built hybrid search index for a single protocol document.

    Build once per pipeline run (in pipeline._build_engine()), then pass
    to RAGExtractionEngine which calls retrieve() for each ICF section.

    Usage::

        index = ProtocolIndex(protocol, config, embedding_client)
        index.build()                        # chunks + embeds the protocol
        chunks = index.retrieve(queries, k)  # multi-query retrieval
        parent = index.get_parent(chunk)     # expand small → parent chunk
    """

    def __init__(
        self,
        protocol: IndexedProtocol,
        config: RAGConfig,
        embedding_client: Any,
        cache_dir: str | None = ".rag_cache",
    ) -> None:
        self.protocol = protocol
        self.config = config
        self._embedding_client = embedding_client
        self._cache_dir = cache_dir or None  # treat empty string as None

        self._small_chunks: list[Chunk] = []
        self._parent_map: dict[str, Chunk] = {}
        self._retriever: HybridRetriever | None = None
        self._parser = DocumentParser(config)

    def build(self) -> None:
        """Parse, chunk, and embed the full protocol.  Call once before retrieve()."""
        t0 = time.time()

        print("[RAG] Parsing protocol into chunks ...")
        self._small_chunks, self._parent_map = self._parser.parse(self.protocol.full_text)

        n_tables = sum(1 for c in self._small_chunks if c.is_table)
        print(
            f"[RAG] {len(self._small_chunks)} small chunks | "
            f"{len(self._parent_map)} parent chunks | "
            f"{n_tables} table region(s) detected."
        )

        # BM25 — instant
        bm25 = BM25Index()
        bm25.build(self._small_chunks)

        # Dense embeddings — served from disk cache when available
        dense = DenseIndex(self.config, self._embedding_client)
        cached_matrix = self._load_embedding_cache()
        if cached_matrix is not None:
            dense._chunks = self._small_chunks
            dense._matrix = cached_matrix
        else:
            n_batches = -(-len(self._small_chunks) // self.config.embedding_batch_size)
            print(
                f"[RAG] Embedding {len(self._small_chunks)} chunks "
                f"({n_batches} API batch(es), model: '{self.config.embedding_model}') ..."
            )
            dense.build(self._small_chunks)
            self._save_embedding_cache(dense._matrix)

        self._retriever = HybridRetriever(bm25, dense, self.config)

        elapsed = time.time() - t0
        print(f"[RAG] Index ready. ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _cache_path(self) -> str | None:
        """Return the cache file path for this protocol + config, or None if disabled."""
        if not self._cache_dir:
            return None
        protocol_id = _extract_protocol_id(self.protocol.source_path)
        fingerprint = _text_fingerprint(self.protocol.full_text)
        model_safe = re.sub(r"[^\w\-]", "_", self.config.embedding_model)
        dims = self.config.embedding_dimensions
        fname = f"{protocol_id}_{model_safe}_{dims}_{fingerprint}.pkl"
        return os.path.join(self._cache_dir, fname)

    def _load_embedding_cache(self) -> np.ndarray | None:
        """Return the cached embedding matrix if it exists and is valid, else None."""
        path = self._cache_path()
        if path is None or not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("n_chunks") != len(self._small_chunks):
                print("[RAG] Embedding cache exists but chunk count changed — re-embedding.")
                return None
            print(f"[RAG] Loaded embeddings from cache -> {path}")
            return data["matrix"]
        except Exception as exc:
            print(f"[RAG] Cache load failed ({exc}); re-embedding.")
            return None

    def _save_embedding_cache(self, matrix: np.ndarray | None) -> None:
        """Persist the embedding matrix to disk for future runs."""
        path = self._cache_path()
        if path is None or matrix is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_chunks": len(self._small_chunks), "matrix": matrix}, f)
        print(f"[RAG] Embedding cache saved -> {path}")

    def retrieve(self, queries: list[str], top_k: int) -> list[Chunk]:
        """Multi-query hybrid retrieval.

        Runs the HybridRetriever for each query independently, then merges
        all results via a second RRF pass and returns the top_k unique chunks.
        This significantly improves recall compared to single-query retrieval.
        """
        assert self._retriever is not None, "Call build() before retrieve()."

        per_query_k = min(top_k * 2, top_k + 30)
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for query in queries:
            results = self._retriever.retrieve(query, per_query_k)
            for rank, chunk in enumerate(results):
                score = 1.0 / (HybridRetriever.RRF_K + rank + 1)
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + score
                chunk_map[chunk.chunk_id] = chunk

        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [chunk_map[cid] for cid in ranked]

    def get_parent(self, chunk: Chunk) -> Chunk:
        """Return the parent chunk for context-rich generation.

        If chunk is already a parent (parent_id is None), returns it unchanged.
        """
        if chunk.parent_id and chunk.parent_id in self._parent_map:
            return self._parent_map[chunk.parent_id]
        return chunk

    def count_tokens(self, text: str) -> int:
        return _count_tokens(text)
