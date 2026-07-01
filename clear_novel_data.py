from __future__ import annotations

import shutil
import sys
from pathlib import Path


TARGET_FOLDERS = (
    Path("summaries"),
    Path("characters"),
    Path("indexes"),
    Path("diagnostics") / "extraction_failures",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python clear_novel_data.py <novel_slug>")
        return 2

    novel_slug = sys.argv[1]
    novel_dir = Path("novels_extracted") / novel_slug
    if not novel_dir.exists():
        print(f"Novel slug does not exist: {novel_slug}")
        return 0
    if not novel_dir.is_dir():
        print(f"Novel path is not a directory: {novel_dir}")
        return 1

    targets = [novel_dir / folder for folder in TARGET_FOLDERS]
    print(f"About to delete generated data for: {novel_slug}")
    for target in targets:
        print(f"- {target}")
    print()
    print("This will not delete chapters/ or metadata.json.")

    answer = input("Continue? (y/n): ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Cancelled.")
        return 0

    deleted_counts: dict[Path, int] = {}
    for target in targets:
        deleted_counts[target] = clear_folder_contents(target)

    print()
    print("Deleted files:")
    for target in targets:
        print(f"- {target}: {deleted_counts[target]}")

    return 0


def clear_folder_contents(folder: Path) -> int:
    if not folder.exists():
        return 0
    if not folder.is_dir():
        return 0

    deleted_files = 0
    for item in folder.iterdir():
        if item.is_dir():
            deleted_files += count_files(item)
            shutil.rmtree(item)
        else:
            item.unlink()
            deleted_files += 1
    return deleted_files


def count_files(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*") if path.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
