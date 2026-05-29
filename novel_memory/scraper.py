from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from .io import read_json, write_json
from .paths import chapter_path, ensure_novel_dirs, novel_dir, slugify


@dataclass(frozen=True)
class Chapter:
    # Local order in your saved dataset.
    # This is always reliable and is used for the filename.
    sequence_number: int

    # Parsed from the visible chapter title if possible.
    # Example: "Chapter 164: ..." -> 164
    # If the title format is weird or has no clear number, this is None.
    chapter_number: int | None

    # Parsed label from the title if possible.
    # Examples: "Chapter 164", "B1 1.4", "Book 1 1.4"
    chapter_label: str | None

    # Full RoyalRoad chapter title from the page header.
    # Example: "Chapter 164: Growing Frustration, And Family"
    title: str

    url: str
    text: str


def chapter_to_json(chapter: Chapter) -> dict[str, object]:
    data = chapter.__dict__.copy()
    data["number"] = chapter.sequence_number
    return data


def fetch_page(url: str, timeout: int = 30):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "novel-memory/0.1"},
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def extract_chapter_text(soup) -> str:
    chapter_content = soup.select_one("div.chapter-inner.chapter-content")
    if chapter_content is None:
        raise ValueError("Could not find RoyalRoad chapter content.")

    title = extract_chapter_title(soup, fallback="").strip()
    paragraphs: list[str] = []

    for paragraph in chapter_content.find_all("p", recursive=False):
        # Preserve line breaks inside system/status blocks.
        for br in paragraph.find_all("br"):
            br.replace_with("\n")

        text = paragraph.get_text(" ", strip=True)
        text = text.replace("\xa0", " ").strip()

        if not text:
            continue

        # Some chapters repeat the title as the first paragraph.
        # Ignore it if it is exactly the same as the page title.
        if title and text == title:
            continue

        paragraphs.append(text)

    if not paragraphs:
        raise ValueError("RoyalRoad chapter content was found, but it contained no paragraph text.")

    return "\n\n".join(paragraphs)


def extract_chapter_title(soup, fallback: str) -> str:
    selectors = (
        ".fic-header h1.font-white.break-word",
        ".fic-header h1",
        ".chapter-title",
        "h1",
    )

    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            title = element.get_text(" ", strip=True)
            title = title.replace("\xa0", " ").strip()
            if title:
                return title

    return fallback


def parse_chapter_metadata(title: str) -> tuple[int | None, str | None]:
    """
    Returns:
        chapter_number:
            Integer chapter number only when it is clearly available.
            Example: "Chapter 164: Title" -> 164

        chapter_label:
            The visible numbering/label part when detected.
            Example: "Chapter 164", "B1 1.4", "Book 1 1.4"

    Important:
        Do not force formats like "B1 1.4" into chapter_number=14.
        Different authors use different numbering systems.
    """
    normalized_title = " ".join(title.replace("\xa0", " ").split())

    numbered_patterns = [
        # Chapter 164: Title
        r"\bchapter\s+(\d+)\b",

        # Ch. 164: Title
        r"\bch\.?\s*(\d+)\b",

        # Book 1 Chapter 4
        r"\bbook\s+\d+\s+chapter\s+(\d+)\b",

        # B1 Chapter 4
        r"\bb\d+\s+chapter\s+(\d+)\b",
    ]

    for pattern in numbered_patterns:
        match = re.search(pattern, normalized_title, flags=re.IGNORECASE)
        if match:
            chapter_number = int(match.group(1))
            chapter_label = match.group(0)
            return chapter_number, chapter_label

    label_only_patterns = [
        # B1 1.4
        r"\bb\d+\s+\d+(?:\.\d+)?\b",

        # Book 1 1.4
        r"\bbook\s+\d+\s+\d+(?:\.\d+)?\b",

        # 1.4 - Title
        r"^\d+(?:\.\d+)?\b",

        # Part Four / Chapter Four / Book Two etc.
        # Kept as label only because converting words to numbers is not always safe.
        r"\b(?:chapter|ch|book|part|arc|episode)\s+[a-z]+\b",
    ]

    for pattern in label_only_patterns:
        match = re.search(pattern, normalized_title, flags=re.IGNORECASE)
        if match:
            return None, match.group(0)

    return None, None


def extract_next_chapter_link(soup, current_url: str) -> str | None:
    for link in soup.find_all("a", href=True):
        label = link.get_text(" ", strip=True).lower()
        if "next chapter" in label:
            return urljoin(current_url, link["href"])
    return None


def existing_chapter_urls(base_dir: Path) -> set[str]:
    urls: set[str] = set()

    for path in sorted((base_dir / "chapters").glob("chapter_*.json")):
        data = read_json(path)
        if data.get("url"):
            urls.add(data["url"])

    return urls


def next_sequence_number(base_dir: Path) -> int:
    numbers: list[int] = []

    for path in (base_dir / "chapters").glob("chapter_*.json"):
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue

    return max(numbers, default=0) + 1


# Backward-compatible alias.
# If other parts of your code still call next_chapter_number(),
# they will continue to work, but internally this is now the local sequence number.
def next_chapter_number(base_dir: Path) -> int:
    return next_sequence_number(base_dir)


def scrape_royalroad(
    title: str,
    start_url: str,
    max_chapters: int | None = None,
    delay_seconds: float = 1.0,
    output_root: Path | str = "novels_extracted",
) -> list[Path]:
    novel_slug = slugify(title)
    base_dir = novel_dir(novel_slug, Path(output_root))
    ensure_novel_dirs(base_dir)

    metadata_path = base_dir / "metadata.json"
    metadata = {
        "title": title,
        "slug": novel_slug,
        "source": "royalroad",
        "start_url": start_url,
    }

    if metadata_path.exists():
        metadata = {**read_json(metadata_path), **metadata}

    write_json(metadata_path, metadata)

    known_urls = existing_chapter_urls(base_dir)
    current_url: str | None = start_url
    sequence_number = next_sequence_number(base_dir)
    saved_paths: list[Path] = []

    while current_url:
        if max_chapters is not None and len(saved_paths) >= max_chapters:
            break

        soup = fetch_page(current_url)
        next_url = extract_next_chapter_link(soup, current_url)

        if current_url not in known_urls:
            fallback_title = f"Chapter {sequence_number}"
            chapter_title = extract_chapter_title(soup, fallback=fallback_title)
            chapter_number, chapter_label = parse_chapter_metadata(chapter_title)

            chapter = Chapter(
                sequence_number=sequence_number,
                chapter_number=chapter_number,
                chapter_label=chapter_label,
                title=chapter_title,
                url=current_url,
                text=extract_chapter_text(soup),
            )

            # Filename uses local scrape order, not the parsed chapter number.
            path = chapter_path(base_dir, sequence_number)
            write_json(path, chapter_to_json(chapter))

            saved_paths.append(path)
            known_urls.add(current_url)
            sequence_number += 1

        current_url = next_url

        if current_url and delay_seconds > 0:
            time.sleep(delay_seconds)

    return saved_paths


def iter_chapter_files(base_dir: Path) -> Iterable[Path]:
    return sorted((base_dir / "chapters").glob("chapter_*.json"))


def migrate_legacy_batches(
    base_dir: Path,
    title: str | None = None,
    force: bool = False,
) -> list[Path]:
    ensure_novel_dirs(base_dir)
    saved_paths: list[Path] = []

    metadata_path = base_dir / "metadata.json"

    if title:
        write_json(
            metadata_path,
            {
                "title": title,
                "slug": base_dir.name,
                "source": "royalroad",
                "legacy_batch_import": True,
            },
        )

    for batch_path in sorted(base_dir.glob("chapters_*.json")):
        entries = read_json(batch_path)
        start_sequence_number = _legacy_start_number(batch_path)

        for offset, entry in enumerate(entries):
            sequence_number = start_sequence_number + offset
            out_path = chapter_path(base_dir, sequence_number)

            if out_path.exists() and not force:
                continue

            text = entry.get("text", "")
            chapter_title = (
                entry.get("title")
                or _title_from_text(text)
                or f"Chapter {sequence_number}"
            )

            chapter_number, chapter_label = parse_chapter_metadata(chapter_title)

            chapter = Chapter(
                sequence_number=sequence_number,
                chapter_number=chapter_number,
                chapter_label=chapter_label,
                title=chapter_title,
                url=entry.get("url", ""),
                text=text,
            )

            write_json(out_path, chapter_to_json(chapter))
            saved_paths.append(out_path)

    return saved_paths


def _legacy_start_number(path: Path) -> int:
    match = re.match(r"chapters_(\d+)_\d+\.json$", path.name)
    if match:
        return int(match.group(1))

    return 1


def _title_from_text(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]

    return None
