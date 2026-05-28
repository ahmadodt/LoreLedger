from pathlib import Path

from novel_memory.paths import character_path, chapter_path, ensure_novel_dirs, slugify


def test_slugify_normalizes_titles():
    assert slugify("A Practical Guide to Evil!") == "a_practical_guide_to_evil"


def test_paths_stay_under_novel_directory(tmp_path: Path):
    ensure_novel_dirs(tmp_path)

    assert chapter_path(tmp_path, 3) == tmp_path / "chapters" / "chapter_0003.json"
    assert character_path(tmp_path, "Arn Ignius") == tmp_path / "characters" / "arn_ignius.json"
    assert (tmp_path / "chapters").is_dir()
