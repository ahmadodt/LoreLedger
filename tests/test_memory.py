from pathlib import Path

import pytest

from novel_memory.memory import character_summary_until, update_character_memory
from novel_memory.paths import ensure_novel_dirs


def test_character_summary_respects_chapter_cutoff(tmp_path: Path):
    ensure_novel_dirs(tmp_path)
    update_character_memory(
        tmp_path,
        {
            "chapter_number": 1,
            "chapter_title": "First",
            "characters": [{"name": "Arn", "aliases": [], "update": "Survives the arena."}],
        },
    )
    update_character_memory(
        tmp_path,
        {
            "chapter_number": 2,
            "chapter_title": "Second",
            "characters": [{"name": "Arn", "aliases": [], "update": "Joins the ludus."}],
        },
    )

    result = character_summary_until(tmp_path, "Arn", 1)

    assert "Survives the arena" in result
    assert "Joins the ludus" not in result


def test_character_summary_errors_for_unknown_character(tmp_path: Path):
    ensure_novel_dirs(tmp_path)

    with pytest.raises(ValueError, match="No character memory"):
        character_summary_until(tmp_path, "Missing", 1)
