from pathlib import Path

from novel_memory.io import read_json, write_json
from novel_memory.scraper import migrate_legacy_batches


def test_migrate_legacy_batches_uses_batch_start_number(tmp_path: Path):
    write_json(
        tmp_path / "chapters_21_22.json",
        [
            {"url": "https://example.test/21", "text": "Twenty One\nBody."},
            {"url": "https://example.test/22", "text": "Twenty Two\nBody."},
        ],
    )

    saved = migrate_legacy_batches(tmp_path, title="Example")

    assert saved == [
        tmp_path / "chapters" / "chapter_0021.json",
        tmp_path / "chapters" / "chapter_0022.json",
    ]
    chapter = read_json(tmp_path / "chapters" / "chapter_0021.json")
    assert chapter["number"] == 21
    assert chapter["title"] == "Twenty One"
