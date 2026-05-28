from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import read_json, write_json
from .memory import update_character_memory
from .paths import chapter_path, ensure_novel_dirs, summary_path
from .scraper import iter_chapter_files


class Summarizer(Protocol):
    def summarize_chapter(self, chapter: dict[str, Any], previous_summary: str | None) -> dict[str, Any]:
        ...


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
    return f"""You summarize fiction chapters for a personal memory tool.

Return only valid JSON inside the requested shape:
{{
  "chapter_summary": "short summary of this chapter",
  "important_events": ["event 1", "event 2"],
  "characters": [
    {{"name": "Character Name", "aliases": ["Optional Alias"], "update": "what changed or was revealed in this chapter"}}
  ]
}}

Previous cumulative summary:
{previous}

Chapter {chapter["number"]}: {chapter["title"]}

Text:
{chapter["text"]}

JSON:
"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("The model did not return JSON.")
    return json.loads(match.group(0))


def normalize_summary(data: dict[str, Any], chapter: dict[str, Any]) -> dict[str, Any]:
    characters = []
    for item in data.get("characters", []):
        name = str(item.get("name", "")).strip()
        update = str(item.get("update", "")).strip()
        if not name or not update:
            continue
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        characters.append({"name": name, "aliases": aliases, "update": update})

    return {
        "chapter_number": int(chapter["number"]),
        "chapter_title": chapter["title"],
        "chapter_url": chapter["url"],
        "chapter_summary": str(data.get("chapter_summary", "")).strip(),
        "important_events": [str(event).strip() for event in data.get("important_events", []) if str(event).strip()],
        "characters": characters,
    }


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
) -> Path:
    ensure_novel_dirs(base_dir)
    in_path = chapter_path(base_dir, chapter_number)
    if not in_path.exists():
        raise FileNotFoundError(f"No chapter file found for chapter {chapter_number}.")

    out_path = summary_path(base_dir, chapter_number)
    if out_path.exists() and not force:
        return out_path

    chapter = read_json(in_path)
    previous_summary = previous_cumulative_summary(base_dir, chapter_number)
    summary = normalize_summary(summarizer.summarize_chapter(chapter, previous_summary), chapter)
    write_json(out_path, summary)
    update_character_memory(base_dir, summary)
    return out_path


def summarize_novel(base_dir: Path, summarizer: Summarizer, force: bool = False) -> list[Path]:
    ensure_novel_dirs(base_dir)
    saved_paths: list[Path] = []
    cumulative_summary: str | None = None

    for chapter_file in iter_chapter_files(base_dir):
        chapter = read_json(chapter_file)
        out_path = summary_path(base_dir, int(chapter["number"]))

        if out_path.exists() and not force:
            summary = read_json(out_path)
        else:
            summary = summarizer.summarize_chapter(chapter, cumulative_summary)
            write_json(out_path, summary)
            update_character_memory(base_dir, summary)
            saved_paths.append(out_path)

        chapter_summary = summary.get("chapter_summary", "")
        if chapter_summary:
            cumulative_summary = (
                f"{cumulative_summary}\nChapter {summary['chapter_number']}: {chapter_summary}"
                if cumulative_summary
                else f"Chapter {summary['chapter_number']}: {chapter_summary}"
            )

    return saved_paths
