import sys
from pathlib import Path
from types import SimpleNamespace

from novel_memory.io import read_json, write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.rag import (
    BM25_INDEX_VERSION,
    EMBEDDING_MODEL_NAME,
    FakeStoryAnswerer,
    LlamaCppStoryAnswerer,
    answer_question,
    build_bm25_index,
    build_embedding_index,
    build_rag_index,
    retrieve_bm25_context,
    retrieve_embedding_context,
    retrieve_hybrid_context,
    retrieve_context,
    _normalize_scores,
)


def test_llama_cpp_story_answerer_passes_all_gpu_layers_value(monkeypatch):
    calls = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    LlamaCppStoryAnswerer(
        model_repo="example/model-GGUF",
        model_file="model.gguf",
        gpu_layers=-1,
    )

    assert calls["n_gpu_layers"] == -1


def test_build_rag_index_uses_summaries_characters_and_chapters(tmp_path: Path):
    _write_fixture_novel(tmp_path)

    path = build_rag_index(tmp_path)

    index = read_json(path)
    source_types = {document["source_type"] for document in index["documents"]}
    assert source_types == {"summary", "character", "chapter"}


def test_build_rag_index_includes_structured_event_details(tmp_path: Path):
    _write_fixture_novel(tmp_path)

    path = build_rag_index(tmp_path)

    index = read_json(path)
    summary_documents = [document for document in index["documents"] if document["id"] == "summary:0002"]
    assert summary_documents
    text = summary_documents[0]["text"]
    assert "Event: Mira warns Arn about the patron." in text
    assert "Type: revelation" in text
    assert "Participants: Mira, Arn" in text
    assert "Evidence: Mira warns him about the patron." in text


def test_build_bm25_index_uses_existing_rag_documents(tmp_path: Path):
    _write_fixture_novel(tmp_path)

    path = build_bm25_index(tmp_path)

    index = read_json(path)
    source_types = {document["source_type"] for document in index["documents"]}
    assert index["version"] == BM25_INDEX_VERSION
    assert source_types == {"summary", "character", "chapter"}
    assert all(isinstance(document["tokens"], list) for document in index["documents"])
    assert any("patron" in document["tokens"] for document in index["documents"])


def test_retrieve_bm25_context_returns_relevant_chapter_context(tmp_path: Path, monkeypatch):
    _write_fixture_novel(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_bm25_okapi", lambda: FakeBM25)
    build_bm25_index(tmp_path)

    contexts = retrieve_bm25_context(tmp_path, "Who warned Arn about the patron?", top_k=2)

    assert contexts
    assert contexts[0].chapter_number == 2
    assert "patron" in contexts[0].text
    assert contexts[0].reference == "Chapter 2 - The Healer"


def test_retrieve_bm25_context_returns_empty_for_empty_index(tmp_path: Path, monkeypatch):
    ensure_novel_dirs(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_bm25_okapi", lambda: FakeBM25)
    build_bm25_index(tmp_path)

    contexts = retrieve_bm25_context(tmp_path, "Who is Nobody?")

    assert contexts == []


def test_build_embedding_index_uses_existing_rag_documents(tmp_path: Path, monkeypatch):
    _write_fixture_novel(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_embedding_model", lambda: FakeEmbeddingModel())

    path = build_embedding_index(tmp_path)

    index = read_json(path)
    source_types = {document["source_type"] for document in index["documents"]}
    assert index["model"] == EMBEDDING_MODEL_NAME
    assert source_types == {"summary", "character", "chapter"}
    assert all(isinstance(document["embedding"], list) for document in index["documents"])
    assert all(document["embedding"] for document in index["documents"])


def test_retrieve_embedding_context_returns_relevant_chapter_context(tmp_path: Path, monkeypatch):
    _write_fixture_novel(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_embedding_model", lambda: FakeEmbeddingModel())
    build_embedding_index(tmp_path)

    contexts = retrieve_embedding_context(tmp_path, "Where is Mira?", top_k=2)

    assert contexts
    assert contexts[0].chapter_number == 2
    assert "Mira" in contexts[0].text
    assert contexts[0].reference == "Chapter 2 - The Healer"


def test_retrieve_embedding_context_returns_empty_for_empty_index(tmp_path: Path, monkeypatch):
    ensure_novel_dirs(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_embedding_model", lambda: FakeEmbeddingModel())
    build_embedding_index(tmp_path)

    contexts = retrieve_embedding_context(tmp_path, "Who is Nobody?")

    assert contexts == []


def test_retrieve_hybrid_context_combines_normalized_scores(tmp_path: Path, monkeypatch):
    _write_fixture_novel(tmp_path)
    build_bm25_index(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_embedding_model", lambda: FakeEmbeddingModel())
    build_embedding_index(tmp_path)
    monkeypatch.setattr(
        "novel_memory.rag._bm25_scores",
        lambda _documents, _query_tokens: {
            "summary:0001": 0.0,
            "summary:0002": 10.0,
            "character:mira:0002": 5.0,
        },
    )
    monkeypatch.setattr(
        "novel_memory.rag._semantic_scores",
        lambda _documents, _question: {
            "summary:0001": 1.0,
            "summary:0002": 0.0,
            "character:mira:0002": 1.0,
        },
    )

    contexts = retrieve_hybrid_context(tmp_path, "Mira patron", top_k=3)

    assert contexts[0].id == "character:mira:0002"
    assert contexts[0].score == 0.75
    assert {context.id: context.score for context in contexts}["summary:0001"] == 0.5
    assert {context.id: context.score for context in contexts}["summary:0002"] == 0.5


def test_retrieve_hybrid_context_returns_relevant_chapter_context(tmp_path: Path, monkeypatch):
    _write_fixture_novel(tmp_path)
    monkeypatch.setattr("novel_memory.rag._load_bm25_okapi", lambda: FakeBM25)
    monkeypatch.setattr("novel_memory.rag._load_embedding_model", lambda: FakeEmbeddingModel())
    build_bm25_index(tmp_path)
    build_embedding_index(tmp_path)

    contexts = retrieve_hybrid_context(tmp_path, "Where is Mira?", top_k=2)

    assert contexts
    assert contexts[0].chapter_number == 2
    assert "Mira" in contexts[0].text
    assert contexts[0].reference == "Chapter 2 - The Healer"


def test_normalize_scores_uses_zero_to_one_range_and_handles_equal_scores():
    normalized = _normalize_scores({"low": 2.0, "mid": 4.0, "high": 6.0})

    assert normalized == {"low": 0.0, "mid": 0.5, "high": 1.0}
    assert _normalize_scores({"left": 3.0, "right": 3.0}) == {"left": 1.0, "right": 1.0}
    assert _normalize_scores({"left": 0.0, "right": 0.0}) == {"left": 0.0, "right": 0.0}


def test_retrieve_context_returns_relevant_character_context(tmp_path: Path):
    _write_fixture_novel(tmp_path)
    build_rag_index(tmp_path)

    contexts = retrieve_context(tmp_path, "What happened to Mira?", top_k=2)

    assert contexts
    assert contexts[0].chapter_number == 2
    assert contexts[0].source_type in {"summary", "character", "chapter"}
    assert "Mira" in contexts[0].text


def test_answer_question_includes_chapter_references(tmp_path: Path):
    _write_fixture_novel(tmp_path)
    build_rag_index(tmp_path)

    result = answer_question(tmp_path, "Who is Arn?", FakeStoryAnswerer(), top_k=2)

    assert "Chapter 1 - The Arena" in result["answer"]
    assert result["references"]


def test_answer_question_reports_missing_context(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    build_rag_index(tmp_path)

    result = answer_question(tmp_path, "Who is Nobody?", FakeStoryAnswerer())

    assert result["answer"] == "I do not have enough stored context to answer that."
    assert result["references"] == []


class FakeEmbeddingModel:
    def encode(self, values, **_kwargs):
        if isinstance(values, str):
            return self._vector(values)
        return [self._vector(value) for value in values]

    def _vector(self, value: str) -> list[float]:
        return [1.0, 0.0] if "mira" in value.lower() else [0.0, 1.0]


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        query = set(query_tokens)
        return [sum(1 for token in document if token in query) for document in self.corpus]


def _write_fixture_novel(base_dir: Path) -> None:
    ensure_novel_dirs(base_dir)
    write_json(
        base_dir / "chapters" / "chapter_0001.json",
        {
            "number": 1,
            "title": "The Arena",
            "url": "https://example.test/1",
            "text": "Arn survives the arena and is taken to a ludus.",
        },
    )
    write_json(
        base_dir / "chapters" / "chapter_0002.json",
        {
            "number": 2,
            "title": "The Healer",
            "url": "https://example.test/2",
            "text": "Mira heals Arn after the duel. Mira warns him about the patron.",
        },
    )
    write_json(
        base_dir / "summaries" / "chapter_0001.json",
        {
            "chapter_number": 1,
            "chapter_title": "The Arena",
            "chapter_url": "https://example.test/1",
            "chapter_summary": {
                "situation": "Arn survives an arena fight.",
                "conflict": "",
                "turning_point": "",
                "consequence": "",
                "hook": "",
            },
            "important_events": ["Arn is taken to a ludus."],
            "characters": [
                {
                    "name": "Arn",
                    "aliases": [],
                    "update": "Survives the arena.",
                }
            ],
        },
    )
    write_json(
        base_dir / "summaries" / "chapter_0002.json",
        {
            "chapter_number": 2,
            "chapter_title": "The Healer",
            "chapter_url": "https://example.test/2",
            "chapter_summary": {
                "situation": "Mira heals Arn and warns him.",
                "conflict": "",
                "turning_point": "",
                "consequence": "",
                "hook": "",
            },
            "important_events": ["Mira warns Arn about the patron."],
            "events": [
                {
                    "description": "Mira warns Arn about the patron.",
                    "event_type": "revelation",
                    "participants": ["Mira", "Arn"],
                    "evidence": "Mira warns him about the patron.",
                }
            ],
            "characters": [
                {
                    "name": "Mira",
                    "aliases": ["The Healer"],
                    "update": "Heals Arn after the duel.",
                }
            ],
        },
    )
    write_json(
        base_dir / "characters" / "arn.json",
        {
            "name": "Arn",
            "aliases": [],
            "timeline": [
                {
                    "chapter_number": 1,
                    "chapter_title": "The Arena",
                    "update": "Survives the arena.",
                }
            ],
        },
    )
    write_json(
        base_dir / "characters" / "mira.json",
        {
            "name": "Mira",
            "aliases": ["The Healer"],
            "timeline": [
                {
                    "chapter_number": 2,
                    "chapter_title": "The Healer",
                    "update": "Heals Arn after the duel.",
                }
            ],
        },
    )
