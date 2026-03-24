"""
Cross-encoder reranking for the RAG extraction backend.

After hybrid retrieval (BM25 + dense + RRF) produces ~20 candidates, a
cross-encoder re-scores each (query, chunk) pair jointly using full
attention across both texts.  This is significantly more accurate than the
bi-encoder cosine similarity used during retrieval, at the cost of running
N forward passes — acceptable here since N ≤ 20 per section.

Implementations
---------------
CrossEncoderReranker
    Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-12-v2.
    ~130 MB model, runs on CPU, ~40-80 ms per (query, passage) pair.
    Downloaded automatically to ~/.cache/huggingface/ on first use.

NoOpReranker
    Pass-through: returns the first top_k candidates unchanged.
    Use --rag-reranker none for ablation studies or faster development runs.

get_reranker(config)
    Factory that instantiates the correct reranker from a RAGConfig.
    Import is deferred so sentence-transformers is only loaded when needed.
"""

from __future__ import annotations

from icf.rag_index import Chunk, RAGConfig

# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Reranks retrieval candidates using a local cross-encoder model.

    The model is loaded lazily on first call to rerank() so pipeline startup
    is not delayed if many sections are skipped or short-circuited.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            print(
                f"[RAG] Loading cross-encoder reranker '{self.model_name}' "
                f"(first use — may download ~130 MB) ..."
            )
            self._model = CrossEncoder(self.model_name)
            print("[RAG] Reranker loaded.")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for the cross-encoder reranker. "
                "Install it with: uv pip install -e '.[rag]'\n"
                "Or use --rag-reranker none to skip reranking."
            ) from e

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Score each (query, chunk.text) pair and return the top_k chunks."""
        if not chunks:
            return []

        self._load()

        pairs = [(query, c.text) for c in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# No-op reranker (ablation / fast mode)
# ---------------------------------------------------------------------------


class NoOpReranker:
    """Returns the first top_k candidates unchanged (no reranking)."""

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_reranker(config: RAGConfig) -> CrossEncoderReranker | NoOpReranker:
    """Instantiate the reranker specified by config.reranker."""
    if config.reranker == "local":
        return CrossEncoderReranker()
    if config.reranker == "none":
        return NoOpReranker()
    raise ValueError(
        f"Unknown reranker: {config.reranker!r}. Choices: 'local', 'none'."
    )
