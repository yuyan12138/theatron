"""
memory_store.py — Episodic memory for the Chameleon trading agent.

Uses ChromaDB as the vector store and sentence-transformers for embeddings.
Memory is disabled by default; ChromaDB and the embedder are only instantiated
when use_memory=True, avoiding heavyweight imports in memory-off runs.

Leakage rule: retrieved records are filtered to dates strictly before the
current trading date, preventing any form of future-data leakage.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Episodic memory store backed by ChromaDB + sentence-transformers.

    Parameters (via config dict)
    ----------------------------
    use_memory : bool
        Master switch. If False, all operations are no-ops.
    memory_top_k : int
        Number of memories to retrieve per query.
    memory_db_path : str
        Directory path for the persistent ChromaDB store.
    symbol : str
    persona : str
    method : str
    model : str
    backend : str
    api_provider : str (optional)
    local_backend : str (optional)
    run_id : int or str
    random_seed : int
    """

    def __init__(self, config: dict):
        self._enabled = bool(config.get("use_memory", False))
        self._top_k = int(config.get("memory_top_k", 1))
        self._collection = None
        self._embedder = None

        if not self._enabled:
            logger.debug("MemoryStore: disabled (use_memory=False).")
            return

        # Derive a unique collection name from the experimental condition
        symbol = config.get("symbol", "UNKNOWN")
        persona = config.get("persona", "unknown")
        method = config.get("method", "zero_shot")
        model = config.get("model", "unknown").replace("/", "_").replace(":", "_")
        backend = config.get("backend", "api")
        api_provider = config.get("api_provider", "openai")
        local_backend = config.get("local_backend", "")
        run_id = config.get("run_id", 1)
        random_seed = config.get("random_seed", 42)

        backend_tag = api_provider if backend == "api" else local_backend
        collection_name = (
            f"{symbol}_{persona}_{method}_{model}_{backend_tag}"
            f"_run{run_id}_seed{random_seed}"
        )
        # ChromaDB collection names must be 3–512 chars, no special chars except _ and -
        collection_name = _sanitize_collection_name(collection_name)

        db_path = config.get("memory_db_path", "./outputs/memory")
        os.makedirs(db_path, exist_ok=True)

        logger.info(
            "MemoryStore: initialising ChromaDB at '%s', collection='%s'",
            db_path,
            collection_name,
        )

        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            chroma_client = chromadb.PersistentClient(path=db_path)
            ef = DefaultEmbeddingFunction()
            self._collection = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            # Store the embedding function to use for query embeddings
            self._embedder = ef
            logger.info("MemoryStore ready (collection='%s').", collection_name)
        except Exception as exc:
            logger.error(
                "MemoryStore initialisation failed: %s. Memory will be disabled.", exc
            )
            self._enabled = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_enabled(self) -> bool:
        """Return True if memory is active and operational."""
        return self._enabled and self._collection is not None

    def add(self, record: dict):
        """
        Store a decision record in the memory collection.

        The embedding is built from:
            "News: {news}\nAction: {action}\nRationale: {rationale}"

        Parameters
        ----------
        record : dict
            Expected keys: date, news (list or str), action, rationale,
            daily_return, cum_return, symbol.
        """
        if not self.is_enabled():
            return

        from src.utils import safe_join_list

        news_text = safe_join_list(record.get("news", []))
        action = str(record.get("action", "HOLD"))
        rationale = str(record.get("rationale", ""))
        date_str = str(record.get("date", ""))

        # Embed on news text only so retrieval is driven by news similarity
        embedding_text = news_text

        try:
            doc_id = f"{date_str}_{action}"

            metadata = {
                "date": date_str,
                "news": news_text[:1000],
                "action": action,
                "rationale": rationale[:500],
                "daily_return": float(record.get("daily_return", 0.0)),
                "cum_return": float(record.get("cum_return", 0.0)),
                "symbol": str(record.get("symbol", "")),
            }

            # Let ChromaDB's embedded function handle embedding via documents=
            self._collection.add(
                ids=[doc_id],
                documents=[embedding_text],
                metadatas=[metadata],
            )
            logger.debug("MemoryStore: stored record for date=%s, action=%s.", date_str, action)
        except Exception as exc:
            logger.warning("MemoryStore.add failed for date=%s: %s", date_str, exc)

    def retrieve(self, query: str, current_date: str) -> list:
        """
        Retrieve the top-k most similar past records whose date is strictly
        before *current_date*.

        Parameters
        ----------
        query : str
            Embedding query string (usually the current day's news text).
        current_date : str
            Today's date string (YYYY-MM-DD). Only records with date < this
            value are returned (leakage prevention).

        Returns
        -------
        list of dict
            Each dict has keys: date, news, action, rationale,
            daily_return, cum_return, symbol.
            Returns an empty list if memory is disabled or no records qualify.
        """
        if not self.is_enabled():
            return []

        try:
            total_docs = self._collection.count()
            if total_docs == 0:
                return []
            n_results = min(max(self._top_k * 5, 10), total_docs)

            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "distances"],
            )

            metadatas = results.get("metadatas", [[]])[0]

            # Filter to dates strictly before current_date
            filtered = [
                m for m in metadatas
                if m.get("date", "") < current_date
            ]

            # Return the top-k after filtering (already ranked by similarity)
            return filtered[: self._top_k]

        except Exception as exc:
            logger.warning("MemoryStore.retrieve failed: %s", exc)
            return []


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _sanitize_collection_name(name: str) -> str:
    """
    Ensure the collection name meets ChromaDB constraints:
    - 3 to 512 characters
    - Only alphanumeric, underscores, and hyphens
    - Cannot start/end with a hyphen
    """
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Trim to 512 chars
    sanitized = sanitized[:512]
    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = sanitized.ljust(3, "_")
    return sanitized
