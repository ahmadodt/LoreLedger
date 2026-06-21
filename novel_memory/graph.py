from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .paths import ensure_novel_dirs


GRAPH_INDEX_PATH = Path("indexes") / "graph.json"


def build_graph(novel_dir: Path) -> Path:
    ensure_novel_dirs(novel_dir)
    graph_path = novel_dir / GRAPH_INDEX_PATH
    summaries = [_summary_with_chapter(path) for path in sorted((novel_dir / "summaries").glob("chapter_*.json"))]
    known_factions = _known_factions(summaries)

    characters: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for chapter_number, summary in summaries:
        for event in summary.get("events", []):
            if str(event.get("event_type", "")).lower() != "relationship":
                continue

            participants = _participants(event)
            if len(participants) < 2:
                continue

            for participant in participants:
                _add_node(characters, participant, chapter_number, known_factions)

            source = participants[0]
            for target in participants[1:]:
                edges.append(
                    {
                        "from": source,
                        "to": target,
                        "relation": _relation_label(str(event.get("description", ""))),
                        "chapter": chapter_number,
                        "description": str(event.get("description", "")).strip(),
                        "evidence": str(event.get("evidence", "")).strip(),
                    }
                )

    write_json(graph_path, {"characters": characters, "edges": edges})
    return graph_path


def query_graph(novel_dir: Path, name: str) -> list[dict[str, Any]]:
    graph_path = novel_dir / GRAPH_INDEX_PATH
    if not graph_path.exists() or not name.strip():
        return []

    needle = name.strip().casefold()
    graph = read_json(graph_path)
    return [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("from", "")).casefold() == needle or str(edge.get("to", "")).casefold() == needle
    ]


def _summary_with_chapter(path: Path) -> tuple[int, dict[str, Any]]:
    summary = read_json(path)
    return int(summary.get("chapter_number", 0)), summary


def _known_factions(summaries: list[tuple[int, dict[str, Any]]]) -> set[str]:
    factions = set()
    for _chapter_number, summary in summaries:
        for event in summary.get("events", []):
            if str(event.get("event_type", "")).lower() != "faction":
                continue
            factions.update(participant.casefold() for participant in _participants(event))
    return factions


def _participants(event: dict[str, Any]) -> list[str]:
    participants = []
    seen = set()
    for participant in event.get("participants", []):
        name = str(participant).strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        participants.append(name)
    return participants


def _add_node(
    characters: dict[str, dict[str, Any]],
    name: str,
    chapter_number: int,
    known_factions: set[str],
) -> None:
    if name not in characters:
        characters[name] = {
            "type": "faction" if name.casefold() in known_factions else "character",
            "first_seen": chapter_number,
            "aliases": [],
        }
        return

    characters[name]["first_seen"] = min(int(characters[name]["first_seen"]), chapter_number)
    if name.casefold() in known_factions:
        characters[name]["type"] = "faction"


def _relation_label(description: str) -> str:
    lowered = description.lower()
    labels = [
        (("saved", "rescued", "protected"), "SAVED"),
        (("betrayed",), "BETRAYED"),
        (("joined", "joins"), "JOINED"),
        (("helped", "aided"), "HELPED"),
        (("warned",), "WARNED"),
        (("healed",), "HEALED"),
        (("met",), "MET"),
        (("allied",), "ALLIED"),
        (("fought", "attacked"), "FOUGHT"),
        (("trusted",), "TRUSTED"),
        (("revealed",), "REVEALED"),
    ]
    for keywords, label in labels:
        if any(keyword in lowered for keyword in keywords):
            return label

    for word in re.findall(r"[A-Za-z]+", description):
        if len(word) > 2 and word.lower() not in {"the", "and", "for", "with"}:
            return word.upper()
    return "RELATED"
