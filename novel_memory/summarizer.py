from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .io import read_json, write_json
from .memory import update_character_memory
from .paths import chapter_path, ensure_novel_dirs, summary_path
from .scraper import iter_chapter_files


class Summarizer(Protocol):
    def summarize_chapter(self, chapter: dict[str, Any], previous_summary: str | None) -> dict[str, Any]:
        ...


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class LlamaCppSummarizer:
    model_repo: str
    model_file: str
    context_size: int = 4096
    gpu_layers: int = 20
    temperature: float = 0.2
    max_tokens: int = 900

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

    def summarize_chapter(self, chapter: dict[str, Any], previous_summary: str | None) -> dict[str, Any]:
        prompt = build_prompt(chapter, previous_summary)
        result = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</json>"],
        )
        text = result["choices"][0]["text"]
        return normalize_summary(parse_json_response(text), chapter)

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        self._llm = None


class FakeSummarizer:
    def summarize_chapter(self, chapter: dict[str, Any], previous_summary: str | None) -> dict[str, Any]:
        words = chapter["text"].split()
        first_sentence = chapter["text"].split(".")[0].strip()
        return normalize_summary(
            {
                "chapter_summary": first_sentence[:300],
                "important_events": [first_sentence[:200]] if first_sentence else [],
                "characters": [
                    {
                        "name": "Arn",
                        "aliases": [],
                        "update": f"Appears in chapter {chapter['number']} with {len(words)} words of source text.",
                    }
                ],
            },
            chapter,
        )


def build_prompt(chapter: dict[str, Any], previous_summary: str | None) -> str:
    previous = previous_summary or "No previous chapter summary is available."
    return f"""You summarize fiction chapters for LoreLedger, a personal story memory and retrieval tool.

Use only the provided chapter text for new facts. Use the previous cumulative summary only for continuity and context.
Do not invent events, motives, names, relationships, powers, or explanations that are not supported by the chapter.

Return strict JSON only. Do not include markdown, comments, prose before the JSON, or prose after the JSON.
Use this exact JSON shape:
{{
  "chapter_summary": "4-8 concise sentences summarizing this chapter",
  "important_events": ["concrete event or state change 1", "concrete event or state change 2"],
  "characters": [
    {{"name": "Character Name", "aliases": ["Optional Alias"], "update": "meaningful character memory update from this chapter"}}
  ]
}}

Guidelines:
- chapter_summary should preserve plot progression, consequences, reveals, decisions, conflicts, and unresolved hooks.
- important_events should contain 3-8 concrete events or state changes that matter after this chapter.
- characters should include every named character whose state meaningfully changes in this chapter.
- Every important event involving a named character must have a matching character update.
- character updates should capture deaths, injuries, discoveries, goals, relationships, secrets, abilities, faction changes, or revelations.
- Keep names, aliases, titles, groups, places, and artifacts as written in the chapter when possible.
- If a field has no supported content, use an empty string or empty array as appropriate.

Previous cumulative summary:
{previous}

Chapter {chapter["number"]}: {chapter["title"]}

Text:
{chapter["text"]}

JSON:
"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError("The model did not return JSON.")

    decoder = json.JSONDecoder()
    data, _end = decoder.raw_decode(stripped[start:])
    if not isinstance(data, dict):
        raise ValueError("The model did not return a JSON object.")
    return data


def normalize_summary(data: dict[str, Any], chapter: dict[str, Any]) -> dict[str, Any]:
    characters = []
    for item in data.get("characters", []):
        name = str(item.get("name", "")).strip()
        update = str(item.get("update", "")).strip()
        if not name or not update:
            continue
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        characters.append({"name": name, "aliases": aliases, "update": update})

    summary = {
        "chapter_number": int(chapter["number"]),
        "chapter_title": chapter["title"],
        "chapter_url": chapter["url"],
        "chapter_summary": str(data.get("chapter_summary", "")).strip(),
        "important_events": [str(event).strip() for event in data.get("important_events", []) if str(event).strip()],
        "characters": characters,
    }
    validate_summary(summary)
    return summary


def validate_summary(summary: dict[str, Any]) -> None:
    important_events = summary.get("important_events", [])
    if summary.get("characters") or not important_events:
        return

    named_events = [event for event in important_events if _mentions_named_character(event)]
    if not named_events:
        return

    raise ValueError(
        "Summary has important events involving named characters but no character updates: "
        + "; ".join(named_events)
    )


def _mentions_named_character(event: Any) -> bool:
    words = re.findall(r"\b[A-Z][a-zA-Z']*\b", str(event))
    ignored = {
        "A",
        "An",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Chapter",
    }
    return any(word not in ignored for word in words)


def previous_cumulative_summary(base_dir: Path, before_chapter: int) -> str | None:
    cumulative_summary: str | None = None

    for path in sorted((base_dir / "summaries").glob("chapter_*.json")):
        summary = read_json(path)
        chapter_number = int(summary.get("chapter_number", 0))
        if chapter_number >= before_chapter:
            continue

        chapter_summary = summary.get("chapter_summary", "")
        if not chapter_summary:
            continue

        line = f"Chapter {chapter_number}: {chapter_summary}"
        cumulative_summary = f"{cumulative_summary}\n{line}" if cumulative_summary else line

    return cumulative_summary


def summarize_chapter(
    base_dir: Path,
    chapter_number: int,
    summarizer: Summarizer,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    ensure_novel_dirs(base_dir)
    in_path = chapter_path(base_dir, chapter_number)
    if not in_path.exists():
        raise FileNotFoundError(f"No chapter file found for chapter {chapter_number}.")

    out_path = summary_path(base_dir, chapter_number)
    if out_path.exists() and not force:
        _emit_progress(progress, "skipped", chapter_number=chapter_number, path=str(out_path))
        return out_path

    _emit_progress(progress, "preparing chapter", chapter_number=chapter_number)
    chapter = read_json(in_path)
    previous_summary = previous_cumulative_summary(base_dir, chapter_number)
    _emit_progress(progress, "generating summary", chapter_number=chapter_number)
    summary = normalize_summary(summarizer.summarize_chapter(chapter, previous_summary), chapter)
    _emit_progress(progress, "saving summary", chapter_number=chapter_number)
    write_json(out_path, summary)
    _emit_progress(progress, "updating character memory", chapter_number=chapter_number)
    update_character_memory(base_dir, summary)
    _emit_progress(progress, "saved", chapter_number=chapter_number, path=str(out_path))
    return out_path


def summarize_novel(base_dir: Path, summarizer: Summarizer, force: bool = False) -> list[Path]:
    return summarize_chapter_range(base_dir, summarizer, force=force)


def summarize_chapter_range(
    base_dir: Path,
    summarizer: Summarizer,
    start_chapter: int | None = None,
    end_chapter: int | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> list[Path]:
    ensure_novel_dirs(base_dir)
    saved_paths: list[Path] = []
    selected_chapters = []

    for chapter_file in iter_chapter_files(base_dir):
        chapter = read_json(chapter_file)
        chapter_number = int(chapter["number"])
        if start_chapter is not None and chapter_number < start_chapter:
            continue
        if end_chapter is not None and chapter_number > end_chapter:
            continue
        selected_chapters.append((chapter_file, chapter))

    if not selected_chapters:
        return saved_paths

    first_chapter = int(selected_chapters[0][1]["number"])
    cumulative_summary = previous_cumulative_summary(base_dir, first_chapter)

    for index, (_chapter_file, chapter) in enumerate(selected_chapters, start=1):
        chapter_number = int(chapter["number"])
        _emit_progress(
            progress,
            "preparing chapter",
            chapter_number=chapter_number,
            completed=index - 1,
            total=len(selected_chapters),
        )
        out_path = summary_path(base_dir, int(chapter["number"]))

        if out_path.exists() and not force:
            summary = read_json(out_path)
            _emit_progress(
                progress,
                "skipped",
                chapter_number=chapter_number,
                completed=index,
                total=len(selected_chapters),
                path=str(out_path),
            )
        else:
            _emit_progress(
                progress,
                "generating summary",
                chapter_number=chapter_number,
                completed=index - 1,
                total=len(selected_chapters),
            )
            summary = normalize_summary(summarizer.summarize_chapter(chapter, cumulative_summary), chapter)
            _emit_progress(
                progress,
                "saving summary",
                chapter_number=chapter_number,
                completed=index - 1,
                total=len(selected_chapters),
            )
            write_json(out_path, summary)
            _emit_progress(
                progress,
                "updating character memory",
                chapter_number=chapter_number,
                completed=index - 1,
                total=len(selected_chapters),
            )
            update_character_memory(base_dir, summary)
            saved_paths.append(out_path)
            _emit_progress(
                progress,
                "saved",
                chapter_number=chapter_number,
                completed=index,
                total=len(selected_chapters),
                path=str(out_path),
            )

        chapter_summary = summary.get("chapter_summary", "")
        if chapter_summary:
            cumulative_summary = (
                f"{cumulative_summary}\nChapter {summary['chapter_number']}: {chapter_summary}"
                if cumulative_summary
                else f"Chapter {summary['chapter_number']}: {chapter_summary}"
            )

    return saved_paths


def _emit_progress(progress: ProgressCallback | None, step: str, **payload: Any) -> None:
    if progress is None:
        return
    progress({"step": step, **payload})
