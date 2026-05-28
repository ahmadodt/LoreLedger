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
    number: int
    title: str
    url: str
    text: str


def fetch_page(url: str, timeout: int = 30):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "novel-memory/0.1"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_chapter_text(soup) -> str:
    chapter_content = soup.find("div", class_="chapter-inner chapter-content")
    if chapter_content is None:
        raise ValueError("Could not find RoyalRoad chapter content.")

    paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in chapter_content.find_all("p")]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
    if not text:
        raise ValueError("RoyalRoad chapter content was found, but it contained no paragraph text.")
    return text


def extract_chapter_title(soup, fallback: str) -> str:
    for selector in ("h1", ".fic-header h1", ".chapter-title"):
        element = soup.select_one(selector)
        if element:
            title = element.get_text(" ", strip=True)
            if title:
                return title
    return fallback


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


def next_chapter_number(base_dir: Path) -> int:
    numbers = []
    for path in (base_dir / "chapters").glob("chapter_*.json"):
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


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
    number = next_chapter_number(base_dir)
    saved_paths: list[Path] = []

    while current_url:
        if max_chapters is not None and len(saved_paths) >= max_chapters:
            break

        soup = fetch_page(current_url)
        next_url = extract_next_chapter_link(soup, current_url)

        if current_url not in known_urls:
            fallback_title = f"Chapter {number}"
            chapter = Chapter(
                number=number,
                title=extract_chapter_title(soup, fallback_title),
                url=current_url,
                text=extract_chapter_text(soup),
            )
            path = chapter_path(base_dir, number)
            write_json(path, chapter.__dict__)
            saved_paths.append(path)
            known_urls.add(current_url)
            number += 1

        current_url = next_url
        if current_url and delay_seconds > 0:
            time.sleep(delay_seconds)

    return saved_paths


def iter_chapter_files(base_dir: Path) -> Iterable[Path]:
    return sorted((base_dir / "chapters").glob("chapter_*.json"))


def migrate_legacy_batches(base_dir: Path, title: str | None = None, force: bool = False) -> list[Path]:
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
        start_number = _legacy_start_number(batch_path)

        for offset, entry in enumerate(entries):
            number = start_number + offset
            out_path = chapter_path(base_dir, number)
            if out_path.exists() and not force:
                continue

            text = entry.get("text", "")
            chapter = Chapter(
                number=number,
                title=entry.get("title") or _title_from_text(text) or f"Chapter {number}",
                url=entry.get("url", ""),
                text=text,
            )
            write_json(out_path, chapter.__dict__)
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
