from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .paths import character_path, slugify


def update_character_memory(base_dir: Path, summary: dict[str, Any]) -> None:
    index: dict[str, Any] = {}
    index_path = base_dir / "indexes" / "characters.json"
    if index_path.exists():
        index = read_json(index_path)

    for character in summary.get("characters", []):
        name = character["name"]
        path, canonical_name, existing_aliases = _resolve_character_identity(base_dir, index, character)
        if path.exists():
            data = read_json(path)
        else:
            data = {"name": canonical_name, "aliases": existing_aliases, "timeline": []}

        aliases = set(data.get("aliases", []))
        aliases.update(character.get("aliases", []))
        if slugify(name) != slugify(data.get("name", name)):
            aliases.add(name)
        timeline = [
            item for item in data.get("timeline", []) if item.get("chapter_number") != summary["chapter_number"]
        ]
        timeline.append(
            {
                "chapter_number": summary["chapter_number"],
                "chapter_title": summary["chapter_title"],
                "update": character["update"],
            }
        )
        timeline.sort(key=lambda item: item["chapter_number"])

        data = {"name": data.get("name", name), "aliases": sorted(aliases), "timeline": timeline}
        write_json(path, data)

        index[slugify(data["name"])] = {
            "name": data["name"],
            "aliases": data["aliases"],
            "path": str(path.relative_to(base_dir)),
        }

    write_json(index_path, index)


def _resolve_character_identity(
    base_dir: Path, index: dict[str, Any], character: dict[str, Any]
) -> tuple[Path, str, list[str]]:
    incoming_names = [character["name"], *character.get("aliases", [])]
    incoming_slugs = {slugify(name) for name in incoming_names if str(name).strip()}

    exact_matches = []
    for item in index.values():
        known_names = [item["name"], *item.get("aliases", [])]
        known_slugs = {slugify(name) for name in known_names if str(name).strip()}
        if incoming_slugs & known_slugs:
            exact_matches.append(item)

    if len(exact_matches) == 1:
        item = exact_matches[0]
        return base_dir / item["path"], item["name"], list(item.get("aliases", []))
    if len(exact_matches) > 1:
        return character_path(base_dir, character["name"]), character["name"], []

    token_matches = []
    incoming_tokens = _name_tokens(character["name"])
    for item in index.values():
        known_names = [item["name"], *item.get("aliases", [])]
        if any(_name_tokens(name) and _name_tokens(name).issubset(incoming_tokens) for name in known_names):
            token_matches.append(item)

    if len(token_matches) == 1:
        item = token_matches[0]
        return base_dir / item["path"], item["name"], list(item.get("aliases", []))

    return character_path(base_dir, character["name"]), character["name"], []


def _name_tokens(name: str) -> set[str]:
    return {part for part in slugify(name).split("_") if part}


def find_character(base_dir: Path, name: str) -> dict[str, Any] | None:
    wanted_slug = slugify(name)
    exact_path = base_dir / "characters" / f"{wanted_slug}.json"
    if exact_path.exists():
        return read_json(exact_path)

    index_path = base_dir / "indexes" / "characters.json"
    if not index_path.exists():
        return None

    index = read_json(index_path)
    for item in index.values():
        names = [item["name"], *item.get("aliases", [])]
        if any(slugify(candidate) == wanted_slug for candidate in names):
            return read_json(base_dir / item["path"])
    return None


def character_summary_until(base_dir: Path, name: str, chapter_number: int) -> str:
    character = find_character(base_dir, name)
    if character is None:
        raise ValueError(f"No character memory found for {name!r}.")

    entries = [
        item for item in character.get("timeline", []) if int(item["chapter_number"]) <= chapter_number
    ]
    if not entries:
        raise ValueError(f"No memory for {character['name']!r} at or before chapter {chapter_number}.")

    lines = [f"{character['name']} through chapter {chapter_number}:"]
    for item in entries:
        lines.append(f"- Chapter {item['chapter_number']} ({item['chapter_title']}): {item['update']}")
    return "\n".join(lines)
