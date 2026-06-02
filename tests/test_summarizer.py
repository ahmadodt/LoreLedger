from pathlib import Path
from types import SimpleNamespace

import sys
import time

import pytest

from novel_memory import power
from novel_memory.io import read_json, write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.summarization_jobs import get_summarization_status, start_summarization_job
from novel_memory.summarizer import (
    FakeSummarizer,
    LlamaCppSummarizer,
    build_prompt,
    parse_json_response,
    summarize_chapter,
    summarize_chapter_range,
    summarize_novel,
)


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


def test_parse_json_response_allows_trailing_model_text():
    parsed = parse_json_response(
        """
        {
          "chapter_summary": "Chloe wakes up.",
          "important_events": ["Chloe wakes up."],
          "characters": []
        }
        Here is the requested summary.
        """
    )

    assert parsed["chapter_summary"] == "Chloe wakes up."


def test_build_prompt_guides_strict_story_memory_summary():
    prompt = build_prompt(
        {
            "number": 7,
            "title": "The Gate",
            "url": "https://example.test/7",
            "text": "Mira reveals the gate key is broken.",
        },
        "Chapter 6: Mira finds the gate key.",
    )

    assert "Return strict JSON only" in prompt
    assert "Use only the provided chapter text for new facts" in prompt
    assert "previous cumulative summary only for continuity" in prompt
    assert "4-8 concise sentences" in prompt
    assert "3-8 concrete events or state changes" in prompt
    assert "status changes, goals, relationships, secrets, injuries, abilities, faction changes, or revelations" in prompt
    assert "Chapter 7: The Gate" in prompt
    assert "Chapter 6: Mira finds the gate key." in prompt


def test_summarize_chapter_range_skips_existing_by_default(tmp_path: Path):
    _write_chapters(tmp_path, count=3)
    write_json(
        tmp_path / "summaries" / "chapter_0002.json",
        {
            "chapter_number": 2,
            "chapter_title": "Chapter 2",
            "chapter_url": "https://example.test/2",
            "chapter_summary": "Already summarized.",
            "important_events": [],
            "characters": [],
        },
    )
    events = []

    saved = summarize_chapter_range(
        tmp_path,
        FakeSummarizer(),
        start_chapter=2,
        end_chapter=3,
        progress=events.append,
    )

    assert saved == [tmp_path / "summaries" / "chapter_0003.json"]
    assert read_json(tmp_path / "summaries" / "chapter_0002.json")["chapter_summary"] == "Already summarized."
    assert any(event["step"] == "skipped" and event["chapter_number"] == 2 for event in events)
    assert any(event["step"] == "saved" and event["chapter_number"] == 3 for event in events)


def test_summarize_chapter_range_regenerates_when_forced(tmp_path: Path):
    _write_chapters(tmp_path, count=2)
    write_json(
        tmp_path / "summaries" / "chapter_0002.json",
        {
            "chapter_number": 2,
            "chapter_title": "Chapter 2",
            "chapter_url": "https://example.test/2",
            "chapter_summary": "Old summary.",
            "important_events": [],
            "characters": [],
        },
    )

    saved = summarize_chapter_range(tmp_path, FakeSummarizer(), start_chapter=2, end_chapter=2, force=True)

    assert saved == [tmp_path / "summaries" / "chapter_0002.json"]
    assert read_json(tmp_path / "summaries" / "chapter_0002.json")["chapter_summary"] == "Arn does thing 2"
    assert (tmp_path / "characters" / "arn.json").exists()


def test_llama_cpp_summarizer_close_releases_model(monkeypatch):
    calls = {"closed": False}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def close(self):
            calls["closed"] = True

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    summarizer = LlamaCppSummarizer(model_repo="example/model-GGUF", model_file="*Q4_K_M.gguf")
    summarizer.close()

    assert calls["closed"] is True


def test_prevent_system_sleep_uses_windows_execution_state(monkeypatch):
    calls = []
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power, "_set_thread_execution_state", calls.append)

    with power.prevent_system_sleep():
        assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]

    assert calls == [
        power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED,
        power.ES_CONTINUOUS,
    ]


def test_prevent_system_sleep_restores_after_error(monkeypatch):
    calls = []
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power, "_set_thread_execution_state", calls.append)

    with pytest.raises(RuntimeError):
        with power.prevent_system_sleep():
            raise RuntimeError("model failed")

    assert calls[-1] == power.ES_CONTINUOUS


def test_background_summarization_job_closes_model(tmp_path: Path, monkeypatch):
    _write_chapters(tmp_path, count=1)
    instances = []

    class ClosingSummarizer:
        def __init__(self, **_kwargs):
            self.closed = False
            instances.append(self)

        def summarize_chapter(self, chapter, previous_summary):
            return {
                "chapter_summary": f"Summary for {chapter['number']}",
                "important_events": [],
                "characters": [],
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr("novel_memory.summarization_jobs.LlamaCppSummarizer", ClosingSummarizer)

    status = start_summarization_job(
        tmp_path,
        novel_slug="example",
        model_config={
            "model_repo": "example/model-GGUF",
            "model_file": "*Q4_K_M.gguf",
            "context_size": 2048,
            "gpu_layers": 0,
            "temperature": 0.2,
        },
        start_chapter=1,
        end_chapter=1,
    )

    for _ in range(50):
        status = get_summarization_status(tmp_path) or status
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "finished"
    assert instances and instances[0].closed is True


def _write_chapters(base_dir: Path, count: int) -> None:
    ensure_novel_dirs(base_dir)
    for number in range(1, count + 1):
        write_json(
            base_dir / "chapters" / f"chapter_{number:04d}.json",
            {
                "number": number,
                "title": f"Chapter {number}",
                "url": f"https://example.test/{number}",
                "text": f"Arn does thing {number}. Then the chapter ends.",
            },
        )
