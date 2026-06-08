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
    ExtractionAttemptsError,
    FakeSummarizer,
    LlamaCppSummarizer,
    PREVIOUS_SUMMARY_LIMIT,
    build_prompt,
    normalize_summary,
    parse_json_response,
    previous_cumulative_summary,
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


def test_previous_cumulative_summary_uses_only_five_latest_summaries(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    for number in range(1, 8):
        write_json(
            tmp_path / "summaries" / f"chapter_{number:04d}.json",
            {
                "chapter_number": number,
                "chapter_title": f"Chapter {number}",
                "chapter_url": f"https://example.test/{number}",
                "chapter_summary": f"Summary {number}",
                "important_events": [],
                "characters": [],
            },
        )

    context = previous_cumulative_summary(tmp_path, before_chapter=8)

    assert context is not None
    assert context.splitlines() == [
        "Chapter 3: Summary 3",
        "Chapter 4: Summary 4",
        "Chapter 5: Summary 5",
        "Chapter 6: Summary 6",
        "Chapter 7: Summary 7",
    ]


def test_previous_cumulative_summary_keeps_all_when_fewer_than_limit(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    for number in range(1, PREVIOUS_SUMMARY_LIMIT):
        write_json(
            tmp_path / "summaries" / f"chapter_{number:04d}.json",
            {
                "chapter_number": number,
                "chapter_title": f"Chapter {number}",
                "chapter_url": f"https://example.test/{number}",
                "chapter_summary": f"Summary {number}",
                "important_events": [],
                "characters": [],
            },
        )

    context = previous_cumulative_summary(tmp_path, before_chapter=PREVIOUS_SUMMARY_LIMIT)

    assert context is not None
    assert len(context.splitlines()) == PREVIOUS_SUMMARY_LIMIT - 1


def test_chapter_range_maintains_rolling_five_summary_context(tmp_path: Path):
    _write_chapters(tmp_path, count=7)
    contexts = []

    class CapturingSummarizer:
        def summarize_chapter(self, chapter, previous_summary):
            contexts.append(previous_summary)
            return {
                "chapter_summary": f"Summary {chapter['number']}",
                "important_events": [],
                "characters": [],
            }

    summarize_chapter_range(tmp_path, CapturingSummarizer())

    assert contexts[0] is None
    assert contexts[5].splitlines() == [
        "Chapter 1: Summary 1",
        "Chapter 2: Summary 2",
        "Chapter 3: Summary 3",
        "Chapter 4: Summary 4",
        "Chapter 5: Summary 5",
    ]
    assert contexts[6].splitlines() == [
        "Chapter 2: Summary 2",
        "Chapter 3: Summary 3",
        "Chapter 4: Summary 4",
        "Chapter 5: Summary 5",
        "Chapter 6: Summary 6",
    ]


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


def test_llama_cpp_summarizer_retries_with_validation_correction(monkeypatch):
    prompts = []
    responses = [
        """{
          "chapter_summary": "Simon dies.",
          "important_events": ["Simon dies from rat bites."],
          "characters": []
        }""",
        """{
          "chapter_summary": "Simon dies.",
          "important_events": ["Simon dies from rat bites."],
          "characters": [{"name": "Simon", "aliases": [], "update": "Dies from rat bites."}]
        }""",
    ]

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def __call__(self, prompt, **_kwargs):
            prompts.append(prompt)
            return {"choices": [{"text": responses.pop(0)}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    summarizer = LlamaCppSummarizer(model_repo="example/model-GGUF", model_file="model.gguf")

    summary = summarizer.summarize_chapter(_chapter(3, "Simon dies from rat bites."), None)

    assert summary["characters"][0]["name"] == "Simon"
    assert len(prompts) == 2
    assert "Correction required after the previous invalid response" in prompts[1]
    assert "no character updates" in prompts[1]


def test_llama_cpp_summarizer_does_not_retry_inference_errors(monkeypatch):
    calls = 0

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def __call__(self, _prompt, **_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("GPU failure")

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    summarizer = LlamaCppSummarizer(model_repo="example/model-GGUF", model_file="model.gguf")

    with pytest.raises(RuntimeError, match="GPU failure"):
        summarizer.summarize_chapter(_chapter(1, "Arn enters."), None)

    assert calls == 1


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
    assert "Every important event involving a named character must have a matching character update" in prompt
    assert "deaths, injuries, discoveries, goals, relationships, secrets, abilities, faction changes, or revelations" in prompt
    assert "Chapter 7: The Gate" in prompt
    assert "Chapter 6: Mira finds the gate key." in prompt


def test_normalize_summary_rejects_named_events_without_character_updates():
    with pytest.raises(ValueError, match="important events involving named characters"):
        normalize_summary(
            {
                "chapter_summary": "Simon finds the dungeon and dies.",
                "important_events": ["Simon dies from rat bites."],
                "characters": [],
            },
            {
                "number": 3,
                "title": "Level One",
                "url": "https://example.test/3",
                "text": "Simon dies from rat bites.",
            },
        )


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


def test_single_chapter_failure_logs_all_model_outputs(tmp_path: Path, monkeypatch):
    ensure_novel_dirs(tmp_path)
    write_json(tmp_path / "chapters" / "chapter_0001.json", _chapter(1, "Arn enters."))
    outputs = ["not json", "still not json", "final invalid output"]

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def __call__(self, _prompt, **_kwargs):
            return {"choices": [{"text": outputs.pop(0)}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    summarizer = LlamaCppSummarizer(model_repo="example/model-GGUF", model_file="model.gguf")

    with pytest.raises(ExtractionAttemptsError, match="3 attempts"):
        summarize_chapter(tmp_path, 1, summarizer)

    failure = read_json(tmp_path / "diagnostics" / "extraction_failures" / "chapter_0001.json")
    assert [attempt["model_output"] for attempt in failure["attempts"]] == [
        "not json",
        "still not json",
        "final invalid output",
    ]
    assert not (tmp_path / "summaries" / "chapter_0001.json").exists()
    assert not (tmp_path / "characters" / "arn.json").exists()


def test_chapter_range_logs_failure_and_continues(tmp_path: Path):
    _write_chapters(tmp_path, count=2)
    events = []

    class PartiallyFailingSummarizer:
        def summarize_chapter(self, chapter, previous_summary):
            if chapter["number"] == 1:
                raise ExtractionAttemptsError(
                    [
                        {
                            "attempt": number,
                            "error_type": "ValueError",
                            "error": "invalid JSON",
                            "model_output": f"bad output {number}",
                        }
                        for number in range(1, 4)
                    ]
                )
            return {
                "chapter_summary": "Arn completes chapter 2.",
                "important_events": ["Arn completes chapter 2."],
                "characters": [{"name": "Arn", "aliases": [], "update": "Completes chapter 2."}],
            }

    saved = summarize_chapter_range(tmp_path, PartiallyFailingSummarizer(), progress=events.append)

    assert saved == [tmp_path / "summaries" / "chapter_0002.json"]
    assert (tmp_path / "diagnostics" / "extraction_failures" / "chapter_0001.json").exists()
    assert any(event["step"] == "failed" and event["chapter_number"] == 1 for event in events)
    assert any(event["step"] == "saved" and event["chapter_number"] == 2 for event in events)


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


def test_background_job_tracks_failed_chapters_and_finishes(tmp_path: Path, monkeypatch):
    _write_chapters(tmp_path, count=2)

    class PartiallyFailingSummarizer:
        def __init__(self, **_kwargs):
            pass

        def summarize_chapter(self, chapter, previous_summary):
            if chapter["number"] == 1:
                raise ExtractionAttemptsError(
                    [{"attempt": 1, "error_type": "ValueError", "error": "bad", "model_output": "bad"}]
                )
            return {
                "chapter_summary": "Summary 2",
                "important_events": [],
                "characters": [],
            }

        def close(self):
            pass

    monkeypatch.setattr("novel_memory.summarization_jobs.LlamaCppSummarizer", PartiallyFailingSummarizer)

    status = start_summarization_job(
        tmp_path,
        novel_slug="example",
        model_config={
            "model_repo": "example/model-GGUF",
            "model_file": "model.gguf",
            "context_size": 2048,
            "gpu_layers": 0,
            "temperature": 0.2,
        },
        start_chapter=1,
        end_chapter=2,
    )

    for _ in range(50):
        status = get_summarization_status(tmp_path) or status
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "finished"
    assert status["failed"] == 1
    assert status["failed_chapters"] == [1]
    assert status["completed"] == 2


def _write_chapters(base_dir: Path, count: int) -> None:
    ensure_novel_dirs(base_dir)
    for number in range(1, count + 1):
        write_json(
            base_dir / "chapters" / f"chapter_{number:04d}.json",
            _chapter(number, f"Arn does thing {number}. Then the chapter ends."),
        )


def _chapter(number: int, text: str) -> dict:
    return {
        "number": number,
        "title": f"Chapter {number}",
        "url": f"https://example.test/{number}",
        "text": text,
    }
