from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import read_json, write_json
from .paths import ensure_novel_dirs
from .scraper import iter_chapter_files


RAG_INDEX_VERSION = 1
RAG_INDEX_PATH = Path("indexes") / "rag.json"


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
        result = self._llm(
            build_answer_prompt(question, contexts),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</answer>"],
        )
        return str(result["choices"][0]["text"]).strip()


class FakeStoryAnswerer:
    def answer_question(self, question: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "I do not have enough stored context to answer that."
        references = ", ".join(context.reference for context in contexts[:2])
        return f"Based on stored context, {question.rstrip('?')} relates to {references}."


def build_rag_index(base_dir: Path, force: bool = False) -> Path:
    ensure_novel_dirs(base_dir)
    index_path = base_dir / RAG_INDEX_PATH
    if index_path.exists() and not force:
        return index_path

    documents = _chapter_summary_documents(base_dir)
    documents.extend(_character_timeline_documents(base_dir))
    documents.extend(_chapter_chunk_documents(base_dir))

    write_json(
        index_path,
        {
            "version": RAG_INDEX_VERSION,
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


def answer_question(
    base_dir: Path,
    question: str,
    answerer: StoryAnswerer,
    top_k: int = 5,
) -> dict[str, Any]:
    contexts = retrieve_context(base_dir, question, top_k=top_k)
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


def _chapter_summary_documents(base_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted((base_dir / "summaries").glob("chapter_*.json")):
        summary = read_json(path)
        chapter_number = int(summary["chapter_number"])
        chapter_title = summary["chapter_title"]
        parts = [summary.get("chapter_summary", "")]
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
