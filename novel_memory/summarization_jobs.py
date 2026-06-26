from __future__ import annotations

import gc
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .paths import ensure_novel_dirs
from .power import prevent_system_sleep
from .scraper import iter_chapter_files
from .summarizer import LlamaCppSummarizer, summarize_chapter_range


STATUS_PATH = Path("indexes") / "summarization_job.json"

_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STATUS: dict[str, Any] | None = None
_CANCEL_REQUESTED = threading.Event()


def start_summarization_job(
    base_dir: Path,
    novel_slug: str,
    model_config: dict[str, Any],
    start_chapter: int,
    end_chapter: int,
    force: bool = False,
) -> dict[str, Any]:
    ensure_novel_dirs(base_dir)
    with _LOCK:
        if _is_thread_active():
            raise RuntimeError("A summarization job is already running.")

        _CANCEL_REQUESTED.clear()
        chapter_numbers = _chapter_numbers_in_range(base_dir, start_chapter, end_chapter)
        if not chapter_numbers:
            raise ValueError("No chapters were found in the selected range.")

        now = _now()
        status = {
            "job_id": uuid.uuid4().hex,
            "base_dir": str(base_dir.resolve()),
            "status": "running",
            "novel": novel_slug,
            "chapter_start": min(chapter_numbers),
            "chapter_end": max(chapter_numbers),
            "current_chapter": None,
            "total": len(chapter_numbers),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "failed_chapters": [],
            "step": "queued",
            "started_at": now,
            "updated_at": now,
            "last_saved_summary": None,
            "error": None,
            "force": bool(force),
            "cancel_requested": False,
        }
        _set_status(base_dir, status)

        thread = threading.Thread(
            target=_run_job,
            args=(base_dir, novel_slug, model_config, status["chapter_start"], status["chapter_end"], force),
            name=f"loreledger-summary-{status['job_id']}",
            daemon=True,
        )
        global _THREAD
        _THREAD = thread
        thread.start()
        return status.copy()


def cancel_summarization_job(base_dir: Path) -> dict[str, Any]:
    with _LOCK:
        status = (_STATUS or _read_status_file(base_dir) or {}).copy()
        if not status or status.get("base_dir") != str(base_dir.resolve()):
            raise RuntimeError("No summarization job was found for this novel.")
        if status.get("status") != "running" or not _is_thread_active():
            raise RuntimeError("No active summarization job is running.")

        _CANCEL_REQUESTED.set()
        status["cancel_requested"] = True
        status["step"] = "cancel requested"
        status["updated_at"] = _now()
        _set_status(base_dir, status)
        return status.copy()


def get_summarization_status(base_dir: Path) -> dict[str, Any] | None:
    with _LOCK:
        if _STATUS is not None and _STATUS.get("base_dir") == str(base_dir.resolve()):
            status = _STATUS.copy()
        else:
            status = _read_status_file(base_dir)

        if status and status.get("status") == "running" and not _is_thread_active():
            status["status"] = "stale"
            status["step"] = "not running"
        return status


def unload_local_models() -> None:
    gc.collect()


def _run_job(
    base_dir: Path,
    novel_slug: str,
    model_config: dict[str, Any],
    start_chapter: int,
    end_chapter: int,
    force: bool,
) -> None:
    summarizer: LlamaCppSummarizer | None = None
    with prevent_system_sleep():
        try:
            _update_status(base_dir, step="loading model")
            summarizer = LlamaCppSummarizer(
                model_repo=model_config["model_repo"],
                model_file=model_config["model_file"],
                context_size=int(model_config["context_size"]),
                gpu_layers=int(model_config["gpu_layers"]),
                temperature=float(model_config["temperature"]),
            )

            def progress(event: dict[str, Any]) -> None:
                updates: dict[str, Any] = {
                    "step": event["step"],
                    "current_chapter": event.get("chapter_number"),
                }
                if "completed" in event:
                    updates["completed"] = event["completed"]
                if event["step"] == "skipped":
                    updates["skipped"] = int((_STATUS or {}).get("skipped", 0)) + 1
                if event["step"] == "saved":
                    updates["last_saved_summary"] = event.get("path")
                if event["step"] == "failed":
                    failed_chapters = list((_STATUS or {}).get("failed_chapters", []))
                    failed_chapters.append(event.get("chapter_number"))
                    updates["failed"] = int((_STATUS or {}).get("failed", 0)) + 1
                    updates["failed_chapters"] = failed_chapters
                if event["step"] == "cancelled":
                    updates["cancel_requested"] = True
                _update_status(base_dir, **updates)

            saved_paths = summarize_chapter_range(
                base_dir,
                summarizer,
                start_chapter=start_chapter,
                end_chapter=end_chapter,
                force=force,
                progress=progress,
                should_cancel=_CANCEL_REQUESTED.is_set,
            )
            if _CANCEL_REQUESTED.is_set():
                _update_status(
                    base_dir,
                    status="cancelled",
                    step="cancelled",
                    last_saved_summary=(
                        str(saved_paths[-1]) if saved_paths else (_STATUS or {}).get("last_saved_summary")
                    ),
                )
                return
            _update_status(
                base_dir,
                status="finished",
                step="finished",
                completed=int((_STATUS or {}).get("total", 0)),
                last_saved_summary=str(saved_paths[-1]) if saved_paths else (_STATUS or {}).get("last_saved_summary"),
            )
        except Exception as exc:
            _update_status(base_dir, status="failed", step="failed", failed=1, error=str(exc))
        finally:
            if summarizer is not None:
                summarizer.close()
            unload_local_models()
            _CANCEL_REQUESTED.clear()


def _chapter_numbers_in_range(base_dir: Path, start_chapter: int, end_chapter: int) -> list[int]:
    numbers = []
    low = min(start_chapter, end_chapter)
    high = max(start_chapter, end_chapter)
    for path in iter_chapter_files(base_dir):
        chapter = read_json(path)
        number = int(chapter["number"])
        if low <= number <= high:
            numbers.append(number)
    return numbers


def _set_status(base_dir: Path, status: dict[str, Any]) -> None:
    global _STATUS
    _STATUS = status.copy()
    write_json(base_dir / STATUS_PATH, _STATUS)


def _update_status(base_dir: Path, **updates: Any) -> None:
    with _LOCK:
        status = (_STATUS or _read_status_file(base_dir) or {}).copy()
        status.update(updates)
        status["updated_at"] = _now()
        _set_status(base_dir, status)


def _read_status_file(base_dir: Path) -> dict[str, Any] | None:
    path = base_dir / STATUS_PATH
    if not path.exists():
        return None
    return read_json(path)


def _is_thread_active() -> bool:
    return _THREAD is not None and _THREAD.is_alive()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(status: dict[str, Any]) -> int:
    started_at = status.get("started_at")
    if not started_at:
        return 0
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return 0
    return max(0, int(time.time() - started.timestamp()))
