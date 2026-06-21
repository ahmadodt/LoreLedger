from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from .graph import build_graph
from .io import read_json, write_json
from .paths import ensure_novel_dirs
from .scraper import iter_chapter_files
from .summarizer import chapter_summary_to_str


RAG_INDEX_VERSION = 1
RAG_INDEX_PATH = Path("indexes") / "rag.json"
BM25_INDEX_VERSION = 1
BM25_INDEX_PATH = Path("indexes") / "bm25.json"
EMBEDDING_INDEX_VERSION = 1
EMBEDDING_INDEX_PATH = Path("indexes") / "embeddings.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_COUNT = 10
RERANK_RESULT_COUNT = 3


@dataclass(frozen=True)
class RetrievedContext:
    id: str
    source_type: str
    chapter_number: int
    chapter_title: str
    text: str
    score: float

    @property
    def reference(self) -> str:
        return f"Chapter {self.chapter_number} - {self.chapter_title}"


class StoryAnswerer(Protocol):
    def answer_question(self, question: str, contexts: list[RetrievedContext]) -> str:
        ...


@dataclass
class LlamaCppStoryAnswerer:
    model_repo: str
    model_file: str
    context_size: int = 4096
    gpu_layers: int = 20
    temperature: float = 0.2
    max_tokens: int = 700

    def __post_init__(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the environment from requirements.txt first."
            ) from exc

        self._llm = Llama.from_pretrained(
            repo_id=self.model_repo,
            filename=self.model_file,
            n_ctx=self.context_size,
            n_gpu_layers=self.gpu_layers,
            verbose=False,
        )

    def answer_question(self, question: str, contexts: list[RetrievedContext]) -> str:
        return self.complete_prompt(build_answer_prompt(question, contexts), stop=["</answer>"])

    def complete_prompt(
        self,
        prompt: str,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        result = self._llm(
            prompt,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return str(result["choices"][0]["text"]).strip()

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        self._llm = None


class FakeStoryAnswerer:
    def answer_question(self, question: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "I do not have enough stored context to answer that."
        references = ", ".join(context.reference for context in contexts[:2])
        return f"Based on stored context, {question.rstrip('?')} relates to {references}."


def build_rag_index(base_dir: Path, force: bool = False) -> Path:
    ensure_novel_dirs(base_dir)
    build_graph(base_dir)
    index_path = base_dir / RAG_INDEX_PATH
    if index_path.exists() and not force:
        return index_path

    documents = _rag_documents(base_dir)

    write_json(
        index_path,
        {
            "version": RAG_INDEX_VERSION,
            "documents": documents,
        },
    )
    return index_path


def build_embedding_index(base_dir: Path, force: bool = False) -> Path:
    ensure_novel_dirs(base_dir)
    build_graph(base_dir)
    index_path = base_dir / EMBEDDING_INDEX_PATH
    if index_path.exists() and not force:
        return index_path

    documents = _rag_documents(base_dir)
    vectors: list[list[float]] = []
    if documents:
        model = _load_embedding_model()
        vectors = [
            _embedding_to_list(vector)
            for vector in model.encode(
                [document["text"] for document in documents],
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        ]

    indexed_documents = [
        {
            **document,
            "embedding": vector,
        }
        for document, vector in zip(documents, vectors)
    ]

    write_json(
        index_path,
        {
            "version": EMBEDDING_INDEX_VERSION,
            "model": EMBEDDING_MODEL_NAME,
            "documents": indexed_documents,
        },
    )
    return index_path


def build_bm25_index(base_dir: Path, force: bool = False) -> Path:
    ensure_novel_dirs(base_dir)
    build_graph(base_dir)
    index_path = base_dir / BM25_INDEX_PATH
    if index_path.exists() and not force:
        return index_path

    documents = [
        {
            **document,
            "tokens": _tokenize(document.get("text", "")),
        }
        for document in _rag_documents(base_dir)
    ]

    write_json(
        index_path,
        {
            "version": BM25_INDEX_VERSION,
            "documents": documents,
        },
    )
    return index_path


def retrieve_context(base_dir: Path, question: str, top_k: int = 5) -> list[RetrievedContext]:
    index_path = base_dir / RAG_INDEX_PATH
    if not index_path.exists():
        build_rag_index(base_dir)

    index = read_json(index_path)
    query_vector = _term_vector(question)
    if not query_vector:
        return []

    contexts = []
    for document in index.get("documents", []):
        score = _cosine_similarity(query_vector, _term_vector(document.get("text", "")))
        score += _recency_boost(question, document)
        if score <= 0:
            continue
        contexts.append(
            RetrievedContext(
                id=document["id"],
                source_type=document["source_type"],
                chapter_number=int(document["chapter_number"]),
                chapter_title=document["chapter_title"],
                text=document["text"],
                score=score,
            )
        )

    contexts.sort(key=lambda item: (-item.score, item.chapter_number, item.id))
    return contexts[:top_k]


def retrieve_bm25_context(base_dir: Path, question: str, top_k: int = 5) -> list[RetrievedContext]:
    index_path = base_dir / BM25_INDEX_PATH
    if not index_path.exists():
        build_bm25_index(base_dir)

    index = read_json(index_path)
    documents = index.get("documents", [])
    query_tokens = _tokenize(question)
    if not documents or not query_tokens:
        return []

    bm25 = _load_bm25_okapi()([document.get("tokens", []) for document in documents])
    scores = bm25.get_scores(query_tokens)

    contexts = []
    for document, score in zip(documents, scores):
        score = float(score)
        if score <= 0:
            continue
        contexts.append(
            RetrievedContext(
                id=document["id"],
                source_type=document["source_type"],
                chapter_number=int(document["chapter_number"]),
                chapter_title=document["chapter_title"],
                text=document["text"],
                score=score,
            )
        )

    contexts.sort(key=lambda item: (-item.score, item.chapter_number, item.id))
    return contexts[:top_k]


def retrieve_hybrid_context(base_dir: Path, question: str, top_k: int = 5) -> list[RetrievedContext]:
    bm25_path = base_dir / BM25_INDEX_PATH
    if not bm25_path.exists():
        build_bm25_index(base_dir)

    embedding_path = base_dir / EMBEDDING_INDEX_PATH
    if not embedding_path.exists():
        build_embedding_index(base_dir)

    bm25_index = read_json(bm25_path)
    embedding_index = read_json(embedding_path)
    bm25_documents = bm25_index.get("documents", [])
    embedding_documents = embedding_index.get("documents", [])
    query_tokens = _tokenize(question)
    if not bm25_documents or not embedding_documents or not query_tokens or not question.strip():
        return []

    bm25_scores = _bm25_scores(bm25_documents, query_tokens)
    semantic_scores = _semantic_scores(embedding_documents, question)
    normalized_bm25 = _normalize_scores(bm25_scores)
    normalized_semantic = _normalize_scores(semantic_scores)

    documents = {document["id"]: document for document in bm25_documents}
    documents.update({document["id"]: document for document in embedding_documents})
    scored_contexts = []
    for document_id, document in documents.items():
        score = (0.5 * normalized_bm25.get(document_id, 0.0)) + (
            0.5 * normalized_semantic.get(document_id, 0.0)
        )
        if score <= 0:
            continue
        scored_contexts.append(
            RetrievedContext(
                id=document["id"],
                source_type=document["source_type"],
                chapter_number=int(document["chapter_number"]),
                chapter_title=document["chapter_title"],
                text=document["text"],
                score=score,
            )
        )

    scored_contexts.sort(key=lambda item: (-item.score, item.chapter_number, item.id))
    return scored_contexts[:top_k]


def retrieve_embedding_context(base_dir: Path, question: str, top_k: int = 5) -> list[RetrievedContext]:
    index_path = base_dir / EMBEDDING_INDEX_PATH
    if not index_path.exists():
        build_embedding_index(base_dir)

    index = read_json(index_path)
    documents = index.get("documents", [])
    if not documents or not question.strip():
        return []

    model = _load_embedding_model()
    query_vector = _embedding_to_list(
        model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    )
    if not query_vector:
        return []

    contexts = []
    for document in documents:
        score = _cosine_similarity_values(query_vector, document.get("embedding", []))
        if score <= 0:
            continue
        contexts.append(
            RetrievedContext(
                id=document["id"],
                source_type=document["source_type"],
                chapter_number=int(document["chapter_number"]),
                chapter_title=document["chapter_title"],
                text=document["text"],
                score=score,
            )
        )

    contexts.sort(key=lambda item: (-item.score, item.chapter_number, item.id))
    return contexts[:top_k]


def answer_question(
    base_dir: Path,
    question: str,
    answerer: StoryAnswerer,
    top_k: int = 5,
    retrieval_mode: str = "tfidf",
    rerank: bool = False,
) -> dict[str, Any]:
    contexts = retrieve_story_context(base_dir, question, top_k=top_k, retrieval_mode=retrieval_mode, rerank=rerank)
    if not contexts:
        return {
            "answer": "I do not have enough stored context to answer that.",
            "references": [],
            "contexts": [],
        }

    answer = answerer.answer_question(question, contexts).strip()
    references = _unique_references(contexts)
    answer = _ensure_references(answer, references)
    return {
        "answer": answer,
        "references": references,
        "contexts": contexts,
    }


def retrieve_story_context(
    base_dir: Path,
    question: str,
    top_k: int = 5,
    retrieval_mode: str = "tfidf",
    rerank: bool = False,
) -> list[RetrievedContext]:
    if rerank:
        contexts = retrieve_story_context(
            base_dir,
            question,
            top_k=RERANK_CANDIDATE_COUNT,
            retrieval_mode=retrieval_mode,
            rerank=False,
        )
        return rerank_contexts(question, contexts, top_k=RERANK_RESULT_COUNT)

    if retrieval_mode == "tfidf":
        return retrieve_context(base_dir, question, top_k=top_k)
    if retrieval_mode == "bm25":
        return retrieve_bm25_context(base_dir, question, top_k=top_k)
    if retrieval_mode == "semantic":
        return retrieve_embedding_context(base_dir, question, top_k=top_k)
    if retrieval_mode == "hybrid":
        return retrieve_hybrid_context(base_dir, question, top_k=top_k)
    raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")


def rerank_contexts(question: str, contexts: list[RetrievedContext], top_k: int = RERANK_RESULT_COUNT) -> list[RetrievedContext]:
    if not contexts or not question.strip():
        return []

    model = _load_reranker_model()
    scores = model.predict([(question, context.text) for context in contexts])
    reranked = [
        RetrievedContext(
            id=context.id,
            source_type=context.source_type,
            chapter_number=context.chapter_number,
            chapter_title=context.chapter_title,
            text=context.text,
            score=float(score),
        )
        for context, score in zip(contexts, scores)
    ]
    reranked.sort(key=lambda item: (-item.score, item.chapter_number, item.id))
    return reranked[:top_k]


def build_answer_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    context_text = "\n\n".join(
        f"[{context.reference} | {context.source_type}]\n{context.text}" for context in contexts
    )
    return f"""You are LoreLedger's story question-answering agent.

Answer the user's question using only the retrieved context below.
If the context is insufficient, say that the stored context is not enough.
Include chapter references in the answer.

Question:
{question}

Retrieved context:
{context_text}

Answer:
"""


def _rag_documents(base_dir: Path) -> list[dict[str, Any]]:
    documents = _chapter_summary_documents(base_dir)
    documents.extend(_character_timeline_documents(base_dir))
    documents.extend(_chapter_chunk_documents(base_dir))
    return documents


def _chapter_summary_documents(base_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted((base_dir / "summaries").glob("chapter_*.json")):
        summary = read_json(path)
        chapter_number = int(summary["chapter_number"])
        chapter_title = summary["chapter_title"]
        parts = [chapter_summary_to_str(summary.get("chapter_summary", {}))]
        event_bits = [_event_document_text(event) for event in summary.get("events", [])]
        if event_bits:
            parts.extend(event_bits)
        else:
            parts.extend(summary.get("important_events", []))
        character_bits = [
            f"{character.get('name', '')}: {character.get('update', '')}"
            for character in summary.get("characters", [])
        ]
        parts.extend(character_bits)
        text = "\n".join(part for part in parts if str(part).strip()).strip()
        if not text:
            continue
        documents.append(
            _document(
                id=f"summary:{chapter_number:04d}",
                source_type="summary",
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                text=text,
            )
        )
    return documents


def _event_document_text(event: dict[str, Any]) -> str:
    description = str(event.get("description", "")).strip()
    if not description:
        return ""
    event_type = str(event.get("event_type", "")).strip()
    participants = ", ".join(str(participant).strip() for participant in event.get("participants", []) if str(participant).strip())
    evidence = str(event.get("evidence", "")).strip()
    parts = [f"Event: {description}"]
    if event_type:
        parts.append(f"Type: {event_type}")
    if participants:
        parts.append(f"Participants: {participants}")
    if evidence:
        parts.append(f"Evidence: {evidence}")
    return ". ".join(parts)


def _character_timeline_documents(base_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted((base_dir / "characters").glob("*.json")):
        character = read_json(path)
        name = character.get("name", path.stem)
        aliases = ", ".join(character.get("aliases", []))
        for item in character.get("timeline", []):
            chapter_number = int(item["chapter_number"])
            chapter_title = item["chapter_title"]
            alias_text = f" Aliases: {aliases}." if aliases else ""
            text = f"{name}.{alias_text} {item.get('update', '')}".strip()
            documents.append(
                _document(
                    id=f"character:{path.stem}:{chapter_number:04d}",
                    source_type="character",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    text=text,
                )
            )
    return documents


def _chapter_chunk_documents(base_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in iter_chapter_files(base_dir):
        chapter = read_json(path)
        chapter_number = int(chapter["number"])
        chapter_title = chapter["title"]
        for index, chunk in enumerate(_chunk_text(chapter.get("text", "")), start=1):
            documents.append(
                _document(
                    id=f"chapter:{chapter_number:04d}:{index:03d}",
                    source_type="chapter",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    text=chunk,
                )
            )
    return documents


def _document(
    id: str,
    source_type: str,
    chapter_number: int,
    chapter_title: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": id,
        "source_type": source_type,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "text": text,
    }


def _chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def _term_vector(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    ]


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _cosine_similarity_values(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _bm25_scores(documents: list[dict[str, Any]], query_tokens: list[str]) -> dict[str, float]:
    bm25 = _load_bm25_okapi()([document.get("tokens", []) for document in documents])
    return {
        document["id"]: float(score)
        for document, score in zip(documents, bm25.get_scores(query_tokens))
    }


def _semantic_scores(documents: list[dict[str, Any]], question: str) -> dict[str, float]:
    model = _load_embedding_model()
    query_vector = _embedding_to_list(
        model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    )
    if not query_vector:
        return {}
    return {
        document["id"]: _cosine_similarity_values(query_vector, document.get("embedding", []))
        for document in documents
    }


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if min_score == max_score:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def _embedding_to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _load_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the environment from requirements.txt first."
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def _load_reranker_model() -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the environment from requirements.txt first."
        ) from exc

    return CrossEncoder(RERANKER_MODEL_NAME, device="cpu")


def _load_bm25_okapi() -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("rank_bm25 is not installed. Install the environment from requirements.txt first.") from exc

    return BM25Okapi


def _recency_boost(question: str, document: dict[str, Any]) -> float:
    tokens = set(_tokenize(question))
    if not {"latest", "recent", "arc"} & tokens:
        return 0.0
    chapter_number = int(document.get("chapter_number", 0))
    return min(chapter_number, 1000) / 100000.0


def _unique_references(contexts: list[RetrievedContext]) -> list[str]:
    references = []
    seen = set()
    for context in contexts:
        if context.reference in seen:
            continue
        seen.add(context.reference)
        references.append(context.reference)
    return references


def _ensure_references(answer: str, references: list[str]) -> str:
    missing = [reference for reference in references if reference not in answer]
    if not missing:
        return answer
    suffix = "\n\nReferences:\n" + "\n".join(f"- {reference}" for reference in references)
    return f"{answer}{suffix}"


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "was",
    "were",
    "are",
    "is",
    "did",
    "does",
    "has",
    "had",
    "have",
    "about",
    "into",
    "from",
    "that",
    "this",
    "there",
    "their",
    "between",
    "relationship",
    "happened",
    "summarize",
    "introduced",
    "chapter",
    "chapters",
}
