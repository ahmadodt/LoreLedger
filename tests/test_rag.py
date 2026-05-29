from pathlib import Path

from novel_memory.io import read_json, write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.rag import FakeStoryAnswerer, answer_question, build_rag_index, retrieve_context


def test_build_rag_index_uses_summaries_characters_and_chapters(tmp_path: Path):
    _write_fixture_novel(tmp_path)

    path = build_rag_index(tmp_path)

    index = read_json(path)
    source_types = {document["source_type"] for document in index["documents"]}
    assert source_types == {"summary", "character", "chapter"}


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
            "chapter_summary": "Arn survives an arena fight.",
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
            "chapter_summary": "Mira heals Arn and warns him.",
            "important_events": ["Mira warns Arn about the patron."],
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
