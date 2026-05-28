from pathlib import Path

from novel_memory.io import write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.summarizer import FakeSummarizer, summarize_novel


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
