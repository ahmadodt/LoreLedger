from __future__ import annotations

import re
from pathlib import Path


OUTPUT_ROOT = Path("novels_extracted")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Cannot create a slug from an empty value.")
    return value


def novel_dir(novel_slug: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / novel_slug


def chapter_path(base_dir: Path, chapter_number: int) -> Path:
    return base_dir / "chapters" / f"chapter_{chapter_number:04d}.json"


def summary_path(base_dir: Path, chapter_number: int) -> Path:
    return base_dir / "summaries" / f"chapter_{chapter_number:04d}.json"


def extraction_failure_path(base_dir: Path, chapter_number: int) -> Path:
    return base_dir / "diagnostics" / "extraction_failures" / f"chapter_{chapter_number:04d}.json"


def character_path(base_dir: Path, character_name: str) -> Path:
    return base_dir / "characters" / f"{slugify(character_name)}.json"


def ensure_novel_dirs(base_dir: Path) -> None:
    for name in ("chapters", "summaries", "characters", "indexes"):
        (base_dir / name).mkdir(parents=True, exist_ok=True)
