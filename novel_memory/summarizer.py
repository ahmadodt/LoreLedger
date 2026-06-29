from __future__ import annotations

import json
import math
import re
import unicodedata
from inspect import Parameter, signature
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .io import read_json, write_json
from .memory import update_character_memory
from .paths import chapter_path, ensure_novel_dirs, extraction_failure_path, summary_path
from .scraper import iter_chapter_files


MAX_EXTRACTION_ATTEMPTS = 3


class Summarizer(Protocol):
    def summarize_chapter(
        self,
        chapter: dict[str, Any],
        previous_summary: str | None,
        max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
    ) -> dict[str, Any]:
        ...


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
PREVIOUS_SUMMARY_LIMIT = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOW_CONFIDENCE_EVIDENCE_THRESHOLD = 0.25
_EVIDENCE_MODEL: Any | None = None
EVENT_TYPES = {
    "action",
    "discovery",
    "decision",
    "relationship",
    "injury",
    "death",
    "ability",
    "faction",
    "revelation",
    "location",
    "other",
}
CHAPTER_SUMMARY_FIELDS = ("situation", "conflict", "turning_point", "consequence", "hook")
CONTINUITY_FLAG_TYPES = {"contradiction", "resolution", "callback"}
MAJOR_CHARACTER_UPDATE_TERMS = {
    "dies",
    "dead",
    "death",
    "injured",
    "injury",
    "wounded",
    "discovers",
    "learns",
    "reveals",
    "revelation",
    "secret",
    "ability",
    "power",
    "faction",
    "relationship",
    "goal",
    "decides",
    "kills",
    "heals",
}


class ExtractionAttemptsError(ValueError):
    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        self.attempts = attempts
        last_error = attempts[-1]["error"] if attempts else "Unknown extraction error."
        super().__init__(f"Chapter extraction failed after {len(attempts)} attempts: {last_error}")


@dataclass
class LlamaCppSummarizer:
    model_repo: str
    model_file: str
    context_size: int = 4096
    gpu_layers: int = 20
    temperature: float = 0.2
    max_tokens: int = 1600

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

    def summarize_chapter(
        self,
        chapter: dict[str, Any],
        previous_summary: str | None,
        max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
    ) -> dict[str, Any]:
        attempts = []
        correction = None

        for attempt_number in range(1, max_attempts + 1):
            prompt = build_prompt(chapter, previous_summary, correction=correction)
            result = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</json>"],
            )
            text = result["choices"][0]["text"]
            try:
                data = parse_json_response(text)
                if "events" not in data:
                    raise ValueError("Field 'events' is required for model summaries.")
                return normalize_summary(data, chapter)
            except ValueError as exc:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "model_output": text,
                    }
                )
                correction = str(exc)

        raise ExtractionAttemptsError(attempts)

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        self._llm = None


class FakeSummarizer:
    def summarize_chapter(
        self,
        chapter: dict[str, Any],
        previous_summary: str | None,
        max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
    ) -> dict[str, Any]:
        words = chapter["text"].split()
        first_sentence = chapter["text"].split(".")[0].strip()
        events = []
        if first_sentence:
            events.append(
                {
                    "description": first_sentence[:200],
                    "event_type": "other",
                    "participants": ["Arn"],
                }
            )
        return normalize_summary(
            {
                "chapter_summary": {
                    "situation": first_sentence[:300],
                    "conflict": "",
                    "turning_point": "",
                    "consequence": "",
                    "hook": "",
                },
                "events": events,
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


def chapter_summary_to_str(chapter_summary: dict[str, Any]) -> str:
    parts = [str(chapter_summary.get(f, "")).strip() for f in CHAPTER_SUMMARY_FIELDS]
    return " ".join(p for p in parts if p)


def build_prompt(
    chapter: dict[str, Any], previous_summary: str | None, correction: str | None = None
) -> str:
    previous = previous_summary or "No previous chapter summary is available."
    correction_text = ""
    if correction:
        correction_text = f"""
Correction required after the previous invalid response:
{correction}
Return the complete JSON again and correct this error. Do not omit any required fields.
"""
    return f"""You summarize fiction chapters for LoreLedger, a personal story memory and retrieval tool.

Use only the provided chapter text for new facts. Use the previous cumulative summary only for continuity and context.
Do not invent events, motives, names, relationships, powers, or explanations that are not supported by the chapter.

Return strict JSON only. Do not include markdown, comments, prose before the JSON, or prose after the JSON.
Use this exact JSON shape:
{{
  "chapter_summary": {{
    "situation": "What state the world or characters are in at the start of this chapter",
    "conflict": "The main tension or problem driving this chapter",
    "turning_point": "The key decision, revelation, or action that changes things",
    "consequence": "What concretely changed as a result",
    "hook": "What is left unresolved or set up for a future chapter"
  }},
  "pov_character": "Name of the point-of-view character, or null if omniscient or unclear",
  "time_skip": "Any time skip mentioned at the start of this chapter, e.g. '3 days later', or null",
  "locations": [
    {{"name": "Location name as written in the chapter", "description": "one sentence description"}}
  ],
  "events": [
    {{"description": "one atomic concrete event or state change", "event_type": "action|discovery|decision|relationship|injury|death|ability|faction|revelation|location|other", "participants": ["Character or entity name"]}}
  ],
  "characters": [
    {{"name": "Character Name", "aliases": ["Optional Alias"], "update": "meaningful character memory update from this chapter"}}
  ],
  "continuity_flags": [
    {{"type": "contradiction|resolution|callback", "description": "what was flagged and why", "evidence": "short exact excerpt from the chapter"}}
  ]
}}

Guidelines:

chapter_summary:
- Each of the five fields should be one to three sentences.
- If a field has no supported content, use an empty string.
- Base every field only on this chapter's text, not the previous summary.

pov_character:
- Use the name as written in the chapter. Use null if the chapter has no clear single POV.

time_skip:
- Only fill this if the chapter explicitly states a time gap. Use null otherwise.

locations:
- Include every named place that features meaningfully in this chapter.
- Description should be brief, one sentence max.

events:
- Include 3 to 10 atomic events or state changes that matter after this chapter.
- Every event must use exactly one event_type from: action, discovery, decision, relationship, injury, death, ability, faction, revelation, location, other.
- Every event must include participants listing named characters, factions, groups, places, or artifacts directly involved.
- Do not infer causation. Finding, witnessing, or learning about an event does not mean the character caused it.
- If a character finds someone already dead, say they found the person dead. Do not claim they killed them.

characters:
- Include every named character whose state meaningfully changes in this chapter.
- Every event involving a named character whose state changes must have a matching character update.
- Character updates should capture deaths, injuries, discoveries, goals, relationships, secrets, abilities, faction changes, or revelations.
- Do not create entries for generic enemies, crowds, or unnamed incidental people unless their change matters beyond this chapter.
- Keep names, aliases, titles, groups, places, and artifacts as written in the chapter.

continuity_flags:
- Compare this chapter against the previous summary and flag anything notable.
- contradiction: this chapter states something that conflicts with established facts.
- resolution: this chapter resolves an unresolved hook or open question from before.
- callback: this chapter references or pays off something established earlier.
- If nothing notable, use an empty array.
{correction_text}

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
    chapter_text = str(chapter.get("text", ""))
    chapter_summary = _normalize_chapter_summary(data)
    events = _normalize_events(data)
    characters = []
    for item in data.get("characters", []):
        name = str(item.get("name", "")).strip()
        update = str(item.get("update", "")).strip()
        if not name or not update:
            continue
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        characters.append({"name": name, "aliases": aliases, "update": update})

    important_events = [event["description"] for event in events]
    if not important_events:
        important_events = [str(event).strip() for event in data.get("important_events", []) if str(event).strip()]

    pov_character = data.get("pov_character")
    pov_character = str(pov_character).strip() if pov_character is not None else None
    time_skip = data.get("time_skip")
    time_skip = str(time_skip).strip() if time_skip is not None else None
    locations = _normalize_locations(data)
    continuity_flags = _normalize_continuity_flags(data, chapter_text)
    _attach_evidence(events, "description", chapter_text)
    _attach_evidence(characters, "update", chapter_text)
    _attach_evidence(locations, "description", chapter_text)

    summary = {
        "chapter_number": int(chapter["number"]),
        "chapter_title": chapter["title"],
        "chapter_url": chapter["url"],
        "chapter_summary": chapter_summary,
        "pov_character": pov_character,
        "time_skip": time_skip,
        "locations": locations,
        "important_events": important_events,
        "events": events,
        "characters": characters,
        "continuity_flags": continuity_flags,
    }
    validate_summary(summary)
    return summary


def find_best_evidence(description: str, chapter_text: str) -> str:
    sentences = _split_sentences(chapter_text)
    if not sentences:
        return ""

    model = _get_evidence_model()
    embeddings = model.encode([description, *sentences], convert_to_numpy=True)
    description_embedding = _as_vector(embeddings[0])
    sentence_embeddings = [_as_vector(embedding) for embedding in embeddings[1:]]

    best_index = 0
    best_score = -1.0
    for index, sentence_embedding in enumerate(sentence_embeddings):
        score = _cosine_similarity(description_embedding, sentence_embedding)
        if score > best_score:
            best_index = index
            best_score = score

    evidence = sentences[best_index]
    if best_score < LOW_CONFIDENCE_EVIDENCE_THRESHOLD:
        return f"LOW_CONFIDENCE: {evidence}"
    return evidence


def _get_evidence_model() -> Any:
    global _EVIDENCE_MODEL
    if _EVIDENCE_MODEL is not None:
        return _EVIDENCE_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the environment from requirements.txt first."
        ) from exc

    _EVIDENCE_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _EVIDENCE_MODEL


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _as_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _attach_evidence(items: list[dict[str, Any]], source_field: str, chapter_text: str) -> None:
    for item in items:
        evidence = find_best_evidence(str(item.get(source_field, "")), chapter_text).strip()
        if not evidence:
            raise RuntimeError("Evidence extraction failed to attach evidence.")
        item["evidence"] = evidence


def _normalize_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("Field 'events' must be an array.")

    for index, item in enumerate(raw_events, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Event {index} must be a JSON object.")

        description = str(item.get("description", "")).strip()
        if not description:
            continue

        event_type = str(item.get("event_type", "")).strip().lower()
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Event {index} field 'event_type' must be one of: {', '.join(sorted(EVENT_TYPES))}."
            )

        raw_participants = item.get("participants", [])
        if not isinstance(raw_participants, list):
            raise ValueError(f"Event {index} field 'participants' must be an array.")
        participants = [str(participant).strip() for participant in raw_participants if str(participant).strip()]
        if _mentions_named_character(description) and not participants:
            raise ValueError(f"Event {index} field 'participants' is required for named-character events.")

        events.append(
            {
                "description": description,
                "event_type": event_type,
                "participants": participants,
            }
        )

    return events


def _normalize_chapter_summary(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("chapter_summary")
    if not isinstance(raw, dict):
        raise ValueError(
            "Field 'chapter_summary' must be a JSON object with fields: "
            "situation, conflict, turning_point, consequence, hook."
        )
    return {field: str(raw.get(field, "")).strip() for field in CHAPTER_SUMMARY_FIELDS}


def _normalize_locations(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("locations", [])
    if not isinstance(raw, list):
        raise ValueError("Field 'locations' must be an array.")
    locations = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description", "")).strip()
        locations.append({"name": name, "description": description})
    return locations


def _normalize_continuity_flags(data: dict[str, Any], _chapter_text: str) -> list[dict[str, Any]]:
    raw = data.get("continuity_flags", [])
    if not isinstance(raw, list):
        raise ValueError("Field 'continuity_flags' must be an array.")
    flags = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        flag_type = str(item.get("type", "")).strip().lower()
        if flag_type not in CONTINUITY_FLAG_TYPES:
            raise ValueError(
                f"Continuity flag field 'type' must be one of: {', '.join(sorted(CONTINUITY_FLAG_TYPES))}."
            )
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        evidence = str(item.get("evidence", "")).strip()
        flags.append({"type": flag_type, "description": description, "evidence": evidence})
    return flags


def _normalize_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def validate_summary(summary: dict[str, Any]) -> None:
    _validate_major_character_updates_have_events(summary)
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


def _validate_major_character_updates_have_events(summary: dict[str, Any]) -> None:
    events = summary.get("events", [])
    if not events:
        return

    for character in summary.get("characters", []):
        update = str(character.get("update", ""))
        if not _is_major_character_update(update):
            continue
        names = [character.get("name", ""), *character.get("aliases", [])]
        if any(_event_matches_character_update(event, names, character.get("evidence", "")) for event in events):
            continue
        raise ValueError(
            "Character update for "
            f"{character.get('name', '')!r} describes a major state change without a related evidence-backed event."
        )


def _is_major_character_update(update: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", update.lower()))
    return bool(tokens & MAJOR_CHARACTER_UPDATE_TERMS)


def _event_matches_character_update(event: dict[str, Any], names: list[Any], character_evidence: Any) -> bool:
    event_text = " ".join(
        [
            str(event.get("description", "")),
            str(event.get("evidence", "")),
            " ".join(str(participant) for participant in event.get("participants", [])),
        ]
    )
    event_slugs = {_name_slug(participant) for participant in event.get("participants", [])}
    for name in names:
        slug = _name_slug(name)
        if slug and slug in event_slugs:
            return True
        if str(name).strip() and _normalize_evidence_text(str(name)) in _normalize_evidence_text(event_text):
            return True

    return _normalize_evidence_text(str(character_evidence)) == _normalize_evidence_text(
        str(event.get("evidence", ""))
    )


def _name_slug(value: Any) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def previous_cumulative_summary(base_dir: Path, before_chapter: int) -> str | None:
    lines = _previous_summary_lines(base_dir, before_chapter)
    return _format_previous_summaries(lines)


def _previous_summary_lines(base_dir: Path, before_chapter: int) -> list[str]:
    lines = []

    for path in sorted((base_dir / "summaries").glob("chapter_*.json")):
        summary = read_json(path)
        chapter_number = int(summary.get("chapter_number", 0))
        if chapter_number >= before_chapter:
            continue

        chapter_summary = chapter_summary_to_str(summary.get("chapter_summary", {}))
        if not chapter_summary:
            continue

        line = f"Chapter {chapter_number}: {chapter_summary}"
        lines.append(line)

    return lines[-PREVIOUS_SUMMARY_LIMIT:]


def _format_previous_summaries(lines: list[str]) -> str | None:
    return "\n".join(lines) if lines else None


def write_extraction_failure(
    base_dir: Path, chapter: dict[str, Any], error: ExtractionAttemptsError
) -> Path:
    path = extraction_failure_path(base_dir, int(chapter["number"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "chapter_number": int(chapter["number"]),
            "chapter_title": chapter["title"],
            "chapter_url": chapter["url"],
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "attempts": error.attempts,
        },
    )
    return path


def summarize_chapter(
    base_dir: Path,
    chapter_number: int,
    summarizer: Summarizer,
    force: bool = False,
    progress: ProgressCallback | None = None,
    max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
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
    try:
        summary = normalize_summary(_run_summarizer(summarizer, chapter, previous_summary, max_attempts), chapter)
    except ExtractionAttemptsError as exc:
        failure_path = write_extraction_failure(base_dir, chapter, exc)
        _emit_progress(
            progress,
            "failed",
            chapter_number=chapter_number,
            error=str(exc),
            path=str(failure_path),
        )
        raise
    _emit_progress(progress, "saving summary", chapter_number=chapter_number)
    write_json(out_path, summary)
    _emit_progress(progress, "updating character memory", chapter_number=chapter_number)
    update_character_memory(base_dir, summary)
    _emit_progress(progress, "saved", chapter_number=chapter_number, path=str(out_path))
    return out_path


def summarize_novel(
    base_dir: Path,
    summarizer: Summarizer,
    force: bool = False,
    max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
) -> list[Path]:
    return summarize_chapter_range(base_dir, summarizer, force=force, max_attempts=max_attempts)


def summarize_chapter_range(
    base_dir: Path,
    summarizer: Summarizer,
    start_chapter: int | None = None,
    end_chapter: int | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
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
    previous_summary_lines = _previous_summary_lines(base_dir, first_chapter)

    for index, (_chapter_file, chapter) in enumerate(selected_chapters, start=1):
        chapter_number = int(chapter["number"])
        if should_cancel is not None and should_cancel():
            _emit_progress(
                progress,
                "cancelled",
                chapter_number=chapter_number,
                completed=index - 1,
                total=len(selected_chapters),
            )
            break

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
            try:
                summary = normalize_summary(
                    _run_summarizer(
                        summarizer,
                        chapter,
                        _format_previous_summaries(previous_summary_lines),
                        max_attempts,
                    ),
                    chapter,
                )
            except ExtractionAttemptsError as exc:
                failure_path = write_extraction_failure(base_dir, chapter, exc)
                _emit_progress(
                    progress,
                    "failed",
                    chapter_number=chapter_number,
                    completed=index,
                    total=len(selected_chapters),
                    error=str(exc),
                    path=str(failure_path),
                )
                continue
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

        chapter_summary_text = chapter_summary_to_str(summary.get("chapter_summary", {}))
        if chapter_summary_text:
            previous_summary_lines.append(f"Chapter {summary['chapter_number']}: {chapter_summary_text}")
            previous_summary_lines = previous_summary_lines[-PREVIOUS_SUMMARY_LIMIT:]

    return saved_paths


def _emit_progress(progress: ProgressCallback | None, step: str, **payload: Any) -> None:
    if progress is None:
        return
    progress({"step": step, **payload})


def _run_summarizer(
    summarizer: Summarizer,
    chapter: dict[str, Any],
    previous_summary: str | None,
    max_attempts: int,
) -> dict[str, Any]:
    method = summarizer.summarize_chapter
    parameters = signature(method).parameters.values()
    accepts_max_attempts = any(
        parameter.name == "max_attempts" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_max_attempts:
        return method(chapter, previous_summary, max_attempts=max_attempts)
    return method(chapter, previous_summary)
