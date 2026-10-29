
"""OpenAI embedding / FAISS semantic retrieval with dataset fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from unicorn_chatbot.data import UnicornRepository


class SemanticRetriever:
    """FAISS retrieval using a consistent OpenAI embedding model."""

    def __init__(
        self,
        repository: UnicornRepository,
        model_name: str,
        index_path: Path,
    ) -> None:
        self.repository = repository
        self.model_name = model_name
        self.index_path = Path(index_path)
        self._model = None
        self._faiss = None
        self._index = None
        self._records: list[dict[str, Any]] = []

    def _dependencies(self):
        """Return FAISS and a replaceable embedding-client factory."""
        import faiss
        from langchain_openai import OpenAIEmbeddings

        return faiss, lambda model_name: OpenAIEmbeddings(model=model_name)

    def _get_embeddings(self):
        if self._model is None:
            _, embedding_factory = self._dependencies()
            self._model = embedding_factory(self.model_name)
        return self._model

    def _dataset_fingerprint(self) -> str:
        serialized = json.dumps(
            self.repository.all_records(),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors using NumPy rather than FAISS native code."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def build(self, force: bool = False) -> None:
        """Build or refresh the index; force=True rebuilds it."""
        faiss, _ = self._dependencies()
        self._faiss = faiss

        metadata_path = self.index_path.with_suffix(".json")
        fingerprint = self._dataset_fingerprint()

        if (
            not force
            and self.index_path.exists()
            and metadata_path.exists()
        ):
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                if (
                    metadata.get("fingerprint") == fingerprint
                    and metadata.get("model_name") == self.model_name
                ):
                    index = faiss.read_index(str(self.index_path))
                    records = metadata["records"]
                    if not isinstance(records, list) or index.d <= 0 or index.ntotal != len(records):
                        raise ValueError("FAISS index and metadata records are inconsistent.")
                    self._index = index
                    self._records = records
                    return
            except (OSError, ValueError, KeyError):
                pass

        records = self.repository.all_records()
        if not records:
            raise ValueError(
                "Cannot build a retrieval index from an empty dataset."
            )

        documents = [
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            for record in records
        ]

        vectors = np.asarray(
            self._get_embeddings().embed_documents(documents),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(records):
            raise ValueError(
                "Embedding provider returned an invalid document vector shape."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Document embeddings contain non-finite values.")

        vectors = self._normalize_vectors(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        metadata_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model_name": self.model_name,
                    "records": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self._index = index
        self._records = records

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return records ranked by cosine similarity."""
        if not query or not query.strip():
            return []

        if top_k is not None:
            k = top_k

        if k <= 0:
            return []

        if min_score is not None and not np.isfinite(min_score):
            raise ValueError("min_score must be a finite similarity threshold.")

        if self._index is None:
            self.build()

        embeddings = self._get_embeddings()
        query_vector = np.asarray(
            [embeddings.embed_query(query.strip())],
            dtype=np.float32,
        )

        if query_vector.ndim != 2:
            raise ValueError("Query embedding must be a two-dimensional vector.")

        if query_vector.shape[1] != self._index.d:
            raise ValueError(
                "Query embedding dimension does not match the FAISS index."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains non-finite values.")

        # Normalize the query with NumPy, matching document normalization.
        query_vector = self._normalize_vectors(query_vector)

        scores, indices = self._index.search(
            query_vector,
            min(k, len(self._records)),
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            score = float(score)

            if min_score is not None and score < min_score:
                continue

            results.append(
                {
                    "record": self._records[int(idx)],
                    "score": score,
                }
            )

        return results
