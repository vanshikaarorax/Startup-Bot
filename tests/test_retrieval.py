
import numpy as np

from unicorn_chatbot.retrieval import SemanticRetriever


class FakeModel:
    def embed_documents(self, texts):
        import numpy as np

        return np.array([
            [1.0, 0.0]
            if "Alpha" in text or "payment" in text
            else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)

    def embed_query(self, text):
        import numpy as np

        return np.array(
            [1.0, 0.0] if "payment" in text else [0.0, 1.0],
            dtype=np.float32,
        )


class FakeIndex:
    def __init__(self):
        self.vectors = None
        self.d = 0

    def add(self, vectors):
        self.vectors = vectors
        self.d = vectors.shape[1]

    def search(self, vector, count):
        import numpy as np

        scores = self.vectors @ vector[0]
        ids = np.argsort(scores)[::-1][:count]
        return scores[ids][None, :], ids[None, :]


def test_retrieval_preserves_source_mapping(repository, tmp_path, monkeypatch):
    retriever = SemanticRetriever(repository, "fake", tmp_path)

    fake_faiss = type("F", (), {
        "IndexFlatIP": staticmethod(lambda _: FakeIndex()),
        "write_index": staticmethod(lambda *_: None),

        # Normalize each vector to unit length.
        "normalize_L2": staticmethod(
            lambda vectors: __import__("faiss").normalize_L2(vectors)
        ),
    })()

    monkeypatch.setattr(
        retriever,
        "_dependencies",
        lambda: (fake_faiss, lambda _: FakeModel()),
    )

    retriever.build(force=True)
    results = retriever.search("payment", top_k=1)

    assert results
    assert results[0]["record"]["company_name"] == "Alpha Pay"
    assert isinstance(results[0]["score"], float)


def test_top_k_threshold_empty_and_invalid_vectors(repository, tmp_path, monkeypatch):
    retriever = SemanticRetriever(repository, "fake", tmp_path)
    fake_faiss = type("F", (), {"IndexFlatIP": staticmethod(lambda _: FakeIndex()), "write_index": staticmethod(lambda *_: None)})()
    monkeypatch.setattr(retriever, "_dependencies", lambda: (fake_faiss, lambda _: FakeModel()))
    retriever.build(force=True)
    assert retriever.search("", top_k=2) == []
    assert len(retriever.search("payment", k=2)) == 2
    assert retriever.search("payment", min_score=1.01) == []
    try:
        retriever.search("payment", min_score=float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc)
    else: raise AssertionError("non-finite threshold was accepted")

    class WrongDimension(FakeModel):
        def embed_query(self, text): return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    retriever._model = WrongDimension()
    try:
        retriever.search("payment")
    except ValueError as exc:
        assert "dimension" in str(exc)
    else: raise AssertionError("invalid vector dimension was accepted")


def test_inconsistent_persisted_metadata_rebuilds(repository, tmp_path, monkeypatch):
    import json
    import faiss
    path = tmp_path / "persistent_index"
    first = SemanticRetriever(repository, "fake", path)
    monkeypatch.setattr(first, "_dependencies", lambda: (faiss, lambda _: FakeModel()))
    first.build(force=True)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["records"] = metadata["records"][:1]
    metadata_path.write_text(json.dumps(metadata))
    rebuilt = SemanticRetriever(repository, "fake", path)
    monkeypatch.setattr(rebuilt, "_dependencies", lambda: (faiss, lambda _: FakeModel()))
    rebuilt.build()
    assert rebuilt._index.ntotal == len(repository.all_records())
    assert len(rebuilt._records) == len(repository.all_records())
