from pathlib import Path
from types import SimpleNamespace

import sys

from novel_memory.io import write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.summarizer import FakeSummarizer, LlamaCppSummarizer, summarize_chapter, summarize_novel


def test_summarize_novel_writes_summary_and_character_memory(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    write_json(
        tmp_path / "chapters" / "chapter_0001.json",
        {
            "number": 1,
            "title": "The Bloodied Eagle",
            "url": "https://example.test/chapter-1",
            "text": "Arn survives the arena. He is taken to a ludus.",
        },
    )

    saved = summarize_novel(tmp_path, FakeSummarizer())

    assert saved == [tmp_path / "summaries" / "chapter_0001.json"]
    assert (tmp_path / "characters" / "arn.json").exists()


def test_summarize_chapter_uses_previous_summary_context(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    write_json(
        tmp_path / "chapters" / "chapter_0001.json",
        {
            "number": 1,
            "title": "First",
            "url": "https://example.test/1",
            "text": "Arn enters the city.",
        },
    )
    write_json(
        tmp_path / "chapters" / "chapter_0002.json",
        {
            "number": 2,
            "title": "Second",
            "url": "https://example.test/2",
            "text": "Arn meets Mira.",
        },
    )
    write_json(
        tmp_path / "summaries" / "chapter_0001.json",
        {
            "chapter_number": 1,
            "chapter_title": "First",
            "chapter_url": "https://example.test/1",
            "chapter_summary": "Arn enters the city.",
            "important_events": [],
            "characters": [],
        },
    )
    calls = {}

    class CapturingSummarizer:
        def summarize_chapter(self, chapter, previous_summary):
            calls["previous_summary"] = previous_summary
            return {
                "chapter_summary": "Arn meets Mira.",
                "important_events": ["Arn meets Mira."],
                "characters": [
                    {
                        "name": "Arn",
                        "aliases": [],
                        "update": "Meets Mira in chapter 2.",
                    }
                ],
            }

    saved = summarize_chapter(tmp_path, 2, CapturingSummarizer())

    assert saved == tmp_path / "summaries" / "chapter_0002.json"
    assert calls["previous_summary"] == "Chapter 1: Arn enters the city."
    assert (tmp_path / "characters" / "arn.json").exists()


def test_llama_cpp_summarizer_loads_hugging_face_gguf(monkeypatch):
    calls = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    LlamaCppSummarizer(
        model_repo="example/model-GGUF",
        model_file="*Q4_K_M.gguf",
        context_size=2048,
        gpu_layers=12,
    )

    assert calls == {
        "repo_id": "example/model-GGUF",
        "filename": "*Q4_K_M.gguf",
        "n_ctx": 2048,
        "n_gpu_layers": 12,
        "verbose": False,
    }
