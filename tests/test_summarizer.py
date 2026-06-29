from pathlib import Path
from types import SimpleNamespace

import sys
import time

import pytest

from novel_memory import power
from novel_memory.io import read_json, write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.summarization_jobs import cancel_summarization_job, get_summarization_status, start_summarization_job
from novel_memory.summarizer import (
    ExtractionAttemptsError,
    FakeSummarizer,
    LlamaCppSummarizer,
    PREVIOUS_SUMMARY_LIMIT,
    build_prompt,
    find_best_evidence,
    normalize_summary,
    parse_json_response,
    previous_cumulative_summary,
    summarize_chapter,
    summarize_chapter_range,
    summarize_novel,
)


@pytest.fixture(autouse=True)
def fake_evidence_model(monkeypatch):
    class FakeEvidenceModel:
        def encode(self, texts, convert_to_numpy=True):
            return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr("novel_memory.summarizer._EVIDENCE_MODEL", FakeEvidenceModel())


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
            "chapter_summary": {
                "situation": "Arn enters the city.",
                "conflict": "",
                "turning_point": "",
                "consequence": "",
                "hook": "",
            },
            "important_events": [],
            "characters": [],
        },
    )
    calls = {}

    class CapturingSummarizer:
        def summarize_chapter(self, chapter, previous_summary):
            calls["previous_summary"] = previous_summary
            return {
                "chapter_summary": {
                    "situation": "Arn meets Mira.",
                    "conflict": "",
                    "turning_point": "",
                    "consequence": "",
                    "hook": "",
                },
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
                "chapter_summary": {
                    "situation": f"Summary {number}",
                    "conflict": "",
                    "turning_point": "",
                    "consequence": "",
                    "hook": "",
                },
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
                "chapter_summary": {
                    "situation": f"Summary {number}",
                    "conflict": "",
                    "turning_point": "",
                    "consequence": "",
                    "hook": "",
                },
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
                "chapter_summary": {
                    "situation": f"Summary {chapter['number']}",
                    "conflict": "",
                    "turning_point": "",
                    "consequence": "",
                    "hook": "",
                },
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


def test_llama_cpp_summarizer_passes_all_gpu_layers_value(monkeypatch):
    calls = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    LlamaCppSummarizer(
        model_repo="example/model-GGUF",
        model_file="model.gguf",
        gpu_layers=-1,
    )

    assert calls["n_gpu_layers"] == -1


def test_llama_cpp_summarizer_retries_with_validation_correction(monkeypatch):
    prompts = []
    responses = [
        """{
          "chapter_summary": {"situation": "Simon dies.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
          "events": [{"description": "Simon dies from rat bites.", "event_type": "fatality", "participants": ["Simon"]}],
          "characters": [{"name": "Simon", "aliases": [], "update": "Dies from rat bites."}]
        }""",
        """{
          "chapter_summary": {"situation": "Simon dies.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
          "events": [{"description": "Simon dies from rat bites.", "event_type": "death", "participants": ["Simon"]}],
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
    assert "field 'event_type' must be one of" in prompts[1]


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
    assert "one to three sentences" in prompt
    assert "3 to 10 atomic events" in prompt
    assert "Every event must use exactly one event_type" in prompt
    assert "Every event must include participants" in prompt
    assert "Every event involving a named character whose state changes must have a matching character update" in prompt
    assert "deaths, injuries, discoveries, goals, relationships, secrets, abilities, faction changes, or revelations" in prompt
    assert "EVIDENCE RULE" not in prompt
    assert "finds someone already dead" in prompt
    assert "Chapter 7: The Gate" in prompt
    assert "Chapter 6: Mira finds the gate key." in prompt


def test_find_best_evidence_returns_matching_sentence(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, convert_to_numpy=True):
            vectors = []
            for text in texts:
                if "gate" in text.lower():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    monkeypatch.setattr("novel_memory.summarizer._EVIDENCE_MODEL", None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    evidence = find_best_evidence(
        "Mira reveals the gate key is broken.",
        "Arn sharpens his blade. Mira reveals the gate key is broken.",
    )

    assert evidence == "Mira reveals the gate key is broken."


def test_find_best_evidence_flags_low_confidence(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, convert_to_numpy=True):
            return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr("novel_memory.summarizer._EVIDENCE_MODEL", None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    evidence = find_best_evidence("A dragon arrives.", "Arn eats breakfast.")

    assert evidence == "LOW_CONFIDENCE: Arn eats breakfast."


def test_find_best_evidence_loads_embedding_model_on_cpu_once(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, convert_to_numpy=True):
            return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr("novel_memory.summarizer._EVIDENCE_MODEL", None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    assert find_best_evidence("Arn enters.", "Arn enters.") == "Arn enters."
    assert find_best_evidence("Arn leaves.", "Arn leaves.") == "Arn leaves."

    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", {"device": "cpu"})]


def test_normalize_summary_attaches_evidence_and_derives_important_events():
    summary = normalize_summary(
        {
            "chapter_summary": {"situation": "Arn meets Mira.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
            "events": [
                {
                    "description": "Arn meets Mira.",
                    "event_type": "relationship",
                    "participants": ["Arn", "Mira"],
                }
            ],
            "characters": [
                {
                    "name": "Arn",
                    "aliases": [],
                    "update": "Meets Mira.",
                }
            ],
        },
        _chapter(1, "Arn meets Mira."),
    )

    assert summary["events"][0]["event_type"] == "relationship"
    assert summary["events"][0]["participants"] == ["Arn", "Mira"]
    assert summary["events"][0]["evidence"] == "Arn meets Mira."
    assert summary["characters"][0]["evidence"] == "Arn meets Mira."
    assert summary["important_events"] == ["Arn meets Mira."]


def test_normalize_summary_raises_runtime_error_when_evidence_attachment_fails(monkeypatch):
    monkeypatch.setattr("novel_memory.summarizer.find_best_evidence", lambda _description, _chapter_text: "")

    with pytest.raises(RuntimeError, match="Evidence extraction failed"):
        normalize_summary(
            {
                "chapter_summary": {"situation": "Arn meets Mira.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
                "events": [
                    {
                        "description": "Arn meets Mira.",
                        "event_type": "relationship",
                        "participants": ["Arn", "Mira"],
                    }
                ],
                "characters": [],
            },
            _chapter(1, "Arn meets Mira."),
        )


def test_normalize_summary_rejects_named_event_without_participants():
    with pytest.raises(ValueError, match="participants"):
        normalize_summary(
            {
                "chapter_summary": {"situation": "Arn meets Mira.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
                "events": [
                    {
                        "description": "Arn meets Mira.",
                        "event_type": "relationship",
                        "participants": [],
                    }
                ],
                "characters": [],
            },
            _chapter(1, "Arn meets Mira."),
        )


def test_normalize_summary_rejects_major_character_update_without_related_event(monkeypatch):
    def fake_find_best_evidence(description, _chapter_text):
        if "Heals" in description:
            return "Mira heals after the duel."
        return "Arn finds a locked door."

    monkeypatch.setattr("novel_memory.summarizer.find_best_evidence", fake_find_best_evidence)

    with pytest.raises(ValueError, match="major state change"):
        normalize_summary(
            {
                "chapter_summary": {"situation": "Arn heals Mira.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
                "events": [
                    {
                        "description": "Arn finds a locked door.",
                        "event_type": "discovery",
                        "participants": ["Arn"],
                    }
                ],
                "characters": [
                    {
                        "name": "Mira",
                        "aliases": [],
                        "update": "Heals after the duel.",
                    }
                ],
            },
            _chapter(1, "Arn finds a locked door. Mira heals after the duel."),
        )


def test_llama_cpp_summarizer_retries_with_event_validation_correction(monkeypatch):
    prompts = []
    responses = [
        """{
          "chapter_summary": {"situation": "Simon dies.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
          "events": [{"description": "Simon dies from rat bites.", "event_type": "death", "participants": "Simon"}],
          "characters": [{"name": "Simon", "aliases": [], "update": "Dies from rat bites."}]
        }""",
        """{
          "chapter_summary": {"situation": "Simon dies.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
          "events": [{"description": "Simon dies from rat bites.", "event_type": "death", "participants": ["Simon"]}],
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

    assert summary["events"][0]["description"] == "Simon dies from rat bites."
    assert len(prompts) == 2
    assert "field 'participants' must be an array" in prompts[1]


def test_normalize_summary_rejects_named_events_without_character_updates():
    with pytest.raises(ValueError, match="important events involving named characters"):
        normalize_summary(
            {
                "chapter_summary": {"situation": "Simon finds the dungeon and dies.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
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


def test_normalize_summary_ignores_model_character_evidence():
    summary = normalize_summary(
        {
            "chapter_summary": {"situation": "Arn meets Mira.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
            "important_events": ["Arn meets Mira."],
            "characters": [
                {
                    "name": "Arn",
                    "aliases": [],
                    "update": "Meets Mira.",
                    "evidence": "Arn defeats Mira.",
                }
            ],
        },
        _chapter(1, "Arn meets Mira."),
    )

    assert summary["characters"][0]["evidence"] == "Arn meets Mira."


def test_normalize_summary_attaches_location_evidence():
    summary = normalize_summary(
        {
            "chapter_summary": {"situation": "Arn enters the tower.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
            "locations": [
                {
                    "name": "Tower",
                    "description": "A tower Arn enters.",
                }
            ],
            "events": [],
            "characters": [],
        },
        _chapter(1, "Arn enters the tower."),
    )

    assert summary["locations"][0]["evidence"] == "Arn enters the tower."


def test_found_dead_update_preserves_source_attribution():
    summary = normalize_summary(
        {
            "chapter_summary": {"situation": "Simon finds the tavern maid dead.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
            "important_events": ["Simon finds the tavern maid already dead."],
            "characters": [
                {
                    "name": "Blonde Tavern Maid",
                    "aliases": [],
                    "update": "Simon finds her already dead.",
                }
            ],
        },
        _chapter(18, "That was when he noticed the body on the floor."),
    )

    assert summary["characters"][0]["update"] == "Simon finds her already dead."


def test_summarize_chapter_range_skips_existing_by_default(tmp_path: Path):
    _write_chapters(tmp_path, count=3)
    write_json(
        tmp_path / "summaries" / "chapter_0002.json",
        {
            "chapter_number": 2,
            "chapter_title": "Chapter 2",
            "chapter_url": "https://example.test/2",
            "chapter_summary": {
                "situation": "Already summarized.",
                "conflict": "",
                "turning_point": "",
                "consequence": "",
                "hook": "",
            },
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
    assert read_json(tmp_path / "summaries" / "chapter_0002.json")["chapter_summary"]["situation"] == "Already summarized."
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
            "chapter_summary": {
                "situation": "Old summary.",
                "conflict": "",
                "turning_point": "",
                "consequence": "",
                "hook": "",
            },
            "important_events": [],
            "characters": [],
        },
    )

    saved = summarize_chapter_range(tmp_path, FakeSummarizer(), start_chapter=2, end_chapter=2, force=True)

    assert saved == [tmp_path / "summaries" / "chapter_0002.json"]
    assert read_json(tmp_path / "summaries" / "chapter_0002.json")["chapter_summary"]["situation"] == "Arn does thing 2"
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
                "chapter_summary": {"situation": "Arn completes chapter 2.", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
                "important_events": ["Arn completes chapter 2."],
                "characters": [
                    {
                        "name": "Arn",
                        "aliases": [],
                        "update": "Completes chapter 2.",
                    }
                ],
            }

    saved = summarize_chapter_range(tmp_path, PartiallyFailingSummarizer(), progress=events.append)

    assert saved == [tmp_path / "summaries" / "chapter_0002.json"]
    assert (tmp_path / "diagnostics" / "extraction_failures" / "chapter_0001.json").exists()
    assert any(event["step"] == "failed" and event["chapter_number"] == 1 for event in events)
    assert any(event["step"] == "saved" and event["chapter_number"] == 2 for event in events)
    summary = read_json(tmp_path / "summaries" / "chapter_0002.json")
    character = read_json(tmp_path / "characters" / "arn.json")
    assert summary["characters"][0]["evidence"] == "Arn does thing 2."
    assert "evidence" not in character["timeline"][0]


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
                "chapter_summary": {"situation": f"Summary for {chapter['number']}", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
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
                "chapter_summary": {"situation": "Summary 2", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
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


def test_background_summarization_job_can_be_cancelled_between_chapters(tmp_path: Path, monkeypatch):
    _write_chapters(tmp_path, count=3)
    instances = []

    class SlowSummarizer:
        def __init__(self, **_kwargs):
            self.closed = False
            instances.append(self)

        def summarize_chapter(self, chapter, previous_summary):
            time.sleep(0.2)
            return {
                "chapter_summary": {"situation": f"Summary {chapter['number']}", "conflict": "", "turning_point": "", "consequence": "", "hook": ""},
                "important_events": [],
                "characters": [],
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr("novel_memory.summarization_jobs.LlamaCppSummarizer", SlowSummarizer)

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
        end_chapter=3,
    )

    for _ in range(50):
        status = get_summarization_status(tmp_path) or status
        if status.get("current_chapter") == 1 and status.get("step") == "generating summary":
            break
        time.sleep(0.02)

    cancel_status = cancel_summarization_job(tmp_path)
    assert cancel_status["cancel_requested"] is True

    for _ in range(50):
        status = get_summarization_status(tmp_path) or status
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "cancelled"
    assert status["completed"] == 1
    assert instances and instances[0].closed is True
    assert (tmp_path / "summaries" / "chapter_0001.json").exists()
    assert not (tmp_path / "summaries" / "chapter_0002.json").exists()


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
