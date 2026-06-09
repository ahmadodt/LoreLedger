import json
from pathlib import Path
from urllib.parse import urlparse


EXAMPLES_PATH = Path(__file__).parents[1] / "examples" / "royalroad_stories.json"


def test_royalroad_story_examples_use_first_chapter_urls():
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

    assert examples
    assert len({example["title"] for example in examples}) == len(examples)

    for example in examples:
        assert set(example) == {"title", "source", "start_url"}
        assert example["title"].strip()
        assert example["source"] == "royalroad"

        parsed = urlparse(example["start_url"])
        assert parsed.scheme == "https"
        assert parsed.netloc == "www.royalroad.com"
        assert "/fiction/" in parsed.path
        assert "/chapter/" in parsed.path
