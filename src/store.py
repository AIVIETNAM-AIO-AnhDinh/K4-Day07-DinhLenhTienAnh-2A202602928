from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.EphemeralClient()
            # Cosine space keeps Chroma's scores on the same scale as the
            # in-memory dot-product path (both are cosine for unit vectors).
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        # doc_id lets delete_document() find every chunk of the same document,
        # even after the same doc id has been added more than once.
        metadata["doc_id"] = doc.id

        record = {
            "id": f"{doc.id}#{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": metadata,
        }
        self._next_index += 1
        return record

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _as_chroma_where(metadata_filter: dict | None) -> dict | None:
        """Chroma needs $and once a filter has more than one key."""
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            return dict(metadata_filter)
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def _chroma_query(self, query: str, top_k: int, where: dict | None) -> list[dict[str, Any]]:
        n_results = min(top_k, self._collection.count())
        if n_results <= 0:
            return []

        response = self._collection.query(
            query_embeddings=[self._embedding_fn(query)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(response["ids"][0]):
            results.append(
                {
                    "id": chunk_id,
                    "content": response["documents"][0][index],
                    "metadata": response["metadatas"][0][index],
                    # cosine distance -> cosine similarity
                    "score": 1.0 - response["distances"][0][index],
                }
            )
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]
        if self._use_chroma:
            self._collection.add(
                ids=[r["id"] for r in records],
                documents=[r["content"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            return self._chroma_query(query, top_k, where=None)
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            return self._chroma_query(query, top_k, where=self._as_chroma_where(metadata_filter))

        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            existing = self._collection.get(where={"doc_id": doc_id}, include=[])
            if not existing["ids"]:
                return False
            self._collection.delete(ids=existing["ids"])
            return True

        remaining = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        if len(remaining) == len(self._store):
            return False
        self._store = remaining
        return True
