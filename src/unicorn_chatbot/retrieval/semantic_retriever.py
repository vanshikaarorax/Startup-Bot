"""OpenAI embedding / FAISS semantic retrieval with dataset fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data import UnicornRepository


class SemanticRetriever:
    def __init__(self, repository: UnicornRepository, model_name: str, index_path: Path) -> None:
        self.repository, self.model_name, self.index_path = repository, model_name, Path(index_path)
        self._model = self._faiss = self._index = None
        self._records: list[dict[str, Any]] = []

    def _dependencies(self):
        import faiss
        from langchain_openai import OpenAIEmbeddings
        return faiss, lambda model_name: OpenAIEmbeddings(model=model_name)

    def _get_embeddings(self):
        if self._model is None:
            _, factory = self._dependencies()
            self._model = factory(self.model_name)
        return self._model

    def _dataset_fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.repository.all_records(), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def build(self, force: bool = False) -> None:
        faiss, _ = self._dependencies()
        self._faiss = faiss
        metadata_path, fingerprint = self.index_path.with_suffix(".json"), self._dataset_fingerprint()
        if not force and self.index_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("fingerprint") == fingerprint and metadata.get("model_name") == self.model_name:
                    index, records = faiss.read_index(str(self.index_path)), metadata["records"]
                    if not isinstance(records, list) or index.d <= 0 or index.ntotal != len(records):
                        raise ValueError("FAISS index and metadata records are inconsistent.")
                    self._index, self._records = index, records
                    return
            except (OSError, ValueError, KeyError):
                pass
        records = self.repository.all_records()
        if not records:
            raise ValueError("Cannot build a retrieval index from an empty dataset.")
        vectors = np.asarray(self._get_embeddings().embed_documents([json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) for record in records]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(records) or not np.isfinite(vectors).all():
            raise ValueError("Embedding provider returned invalid document vectors.")
        vectors = self._normalize_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "model_name": self.model_name, "records": records}, ensure_ascii=False), encoding="utf-8")
        self._index, self._records = index, records

    def search(self, query: str, k: int = 5, *, top_k: int | None = None, min_score: float | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip() or (top_k is not None and top_k <= 0) or k <= 0:
            return []
        if top_k is not None: k = top_k
        if min_score is not None and not np.isfinite(min_score): raise ValueError("min_score must be a finite similarity threshold.")
        if self._index is None: self.build()
        vector = np.asarray([self._get_embeddings().embed_query(query.strip())], dtype=np.float32)
        if vector.ndim != 2: raise ValueError("Query embedding must be a two-dimensional vector.")
        if vector.shape[1] != self._index.d: raise ValueError("Query embedding dimension does not match the FAISS index.")
        if not np.isfinite(vector).all(): raise ValueError("Query embedding contains non-finite values.")
        scores, indices = self._index.search(self._normalize_vectors(vector), min(k, len(self._records)))
        return [{"record": self._records[int(idx)], "score": float(score)} for score, idx in zip(scores[0], indices[0]) if idx >= 0 and (min_score is None or float(score) >= min_score)]
