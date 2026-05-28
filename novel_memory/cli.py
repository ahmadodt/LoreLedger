from __future__ import annotations

import argparse
from pathlib import Path

from .memory import character_summary_until
from .paths import novel_dir
from .scraper import migrate_legacy_batches, scrape_royalroad
from .summarizer import LlamaCppSummarizer, summarize_novel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novel_memory")
    parser.add_argument("--output-root", default="novels_extracted")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="Scrape RoyalRoad chapters.")
    scrape.add_argument("--title", required=True)
    scrape.add_argument("--start-url", required=True)
    scrape.add_argument("--max-chapters", type=int)
    scrape.add_argument("--delay", type=float, default=1.0)

    summarize = subparsers.add_parser("summarize", help="Summarize chapters and update character memory.")
    summarize.add_argument("--novel", required=True, help="Novel slug, for example practical_guide_to_evil.")
    summarize.add_argument("--model-path", required=True)
    summarize.add_argument("--context-size", type=int, default=4096)
    summarize.add_argument("--gpu-layers", type=int, default=20)
    summarize.add_argument("--temperature", type=float, default=0.2)
    summarize.add_argument("--force", action="store_true")

    character = subparsers.add_parser("character", help="Show character memory up to a chapter.")
    character.add_argument("--novel", required=True)
    character.add_argument("--chapter", required=True, type=int)
    character.add_argument("--name", required=True)

    migrate = subparsers.add_parser("migrate-batches", help="Convert old chapters_1_20.json files to per-chapter files.")
    migrate.add_argument("--novel", required=True)
    migrate.add_argument("--title")
    migrate.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)

    if args.command == "scrape":
        saved = scrape_royalroad(
            title=args.title,
            start_url=args.start_url,
            max_chapters=args.max_chapters,
            delay_seconds=args.delay,
            output_root=output_root,
        )
        print(f"Saved {len(saved)} new chapter(s).")
        return

    if args.command == "summarize":
        base_dir = novel_dir(args.novel, output_root)
        summarizer = LlamaCppSummarizer(
            model_path=args.model_path,
            context_size=args.context_size,
            gpu_layers=args.gpu_layers,
            temperature=args.temperature,
        )
        saved = summarize_novel(base_dir, summarizer=summarizer, force=args.force)
        print(f"Saved {len(saved)} new summary file(s).")
        return

    if args.command == "character":
        base_dir = novel_dir(args.novel, output_root)
        print(character_summary_until(base_dir, args.name, args.chapter))
        return

    if args.command == "migrate-batches":
        base_dir = novel_dir(args.novel, output_root)
        saved = migrate_legacy_batches(base_dir, title=args.title, force=args.force)
        print(f"Migrated {len(saved)} chapter file(s).")
        return

    raise AssertionError(f"Unhandled command: {args.command}")
