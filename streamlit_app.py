from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from novel_memory.env import load_project_env
from novel_memory.io import read_json
from novel_memory.memory import character_summary_until
from novel_memory.paths import OUTPUT_ROOT, ensure_novel_dirs, novel_dir
from novel_memory.rag import FakeStoryAnswerer, LlamaCppStoryAnswerer, answer_question, build_rag_index, retrieve_context
from novel_memory.scraper import iter_chapter_files, scrape_royalroad
from novel_memory.summarization_jobs import (
    elapsed_seconds,
    get_summarization_status,
    start_summarization_job,
    unload_local_models,
)
from novel_memory.summarizer import LlamaCppSummarizer


load_project_env()
st.set_page_config(page_title="LoreLedger", page_icon="LL", layout="wide")


def load_novels(output_root: Path = OUTPUT_ROOT) -> list[dict[str, Any]]:
    novels: list[dict[str, Any]] = []
    if not output_root.exists():
        return novels

    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue

        metadata_path = path / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        chapters = list(iter_chapter_files(path))
        summaries = list((path / "summaries").glob("chapter_*.json"))
        characters = list((path / "characters").glob("*.json"))
        if not metadata and not chapters and not summaries and not characters:
            continue

        title = str(metadata.get("title") or path.name.replace("_", " ").title()).strip()
        if not title:
            continue

        novels.append(
            {
                "slug": path.name,
                "title": title,
                "path": path,
                "chapter_count": len(chapters),
                "summary_count": len(summaries),
                "character_count": len(characters),
                "metadata": metadata,
            }
        )

    return novels


def load_chapters(base_dir: Path) -> list[dict[str, Any]]:
    chapters = []
    for path in iter_chapter_files(base_dir):
        chapter = read_json(path)
        number = int(chapter["number"])
        title = str(chapter.get("title") or f"Chapter {number}").strip()
        chapters.append(
            {
                "number": number,
                "title": title,
                "path": path,
                "text": chapter.get("text", ""),
                "url": chapter.get("url", ""),
            }
        )
    return chapters


def load_characters(base_dir: Path) -> list[dict[str, Any]]:
    characters = []
    for path in sorted((base_dir / "characters").glob("*.json")):
        character = read_json(path)
        name = str(character.get("name") or path.stem).strip()
        if not name:
            continue
        characters.append(
            {
                "name": name,
                "aliases": character.get("aliases", []),
                "timeline": character.get("timeline", []),
            }
        )
    return sorted(characters, key=lambda item: item["name"].lower())


def chapter_label(chapter: dict[str, Any]) -> str:
    return f"{chapter['number']:04d} - {chapter['title']}"


def novel_label(novel: dict[str, Any]) -> str:
    return f"{novel['title']} ({novel['chapter_count']} chapters)"


def format_elapsed(seconds: int) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def render_summary(summary: dict[str, Any]) -> None:
    st.success("Summary exists.")
    st.write(summary.get("chapter_summary", ""))
    if summary.get("important_events"):
        st.write("Important events")
        for event in summary["important_events"]:
            st.write(f"- {event}")


def render_job_status(status: dict[str, Any]) -> None:
    state = str(status.get("status", "unknown"))
    total = int(status.get("total") or 0)
    completed = int(status.get("completed") or 0)
    current = status.get("current_chapter")
    step = status.get("step") or "unknown"
    skipped = int(status.get("skipped") or 0)
    failed = int(status.get("failed") or 0)
    elapsed = format_elapsed(elapsed_seconds(status))

    if state == "running":
        st.info(f"Summarization running: {step}")
        st.progress(completed / total if total else 0.0)
    elif state == "finished":
        st.success("Summarization finished.")
    elif state == "failed":
        st.error(f"Summarization failed: {status.get('error')}")
    elif state == "stale":
        st.warning("Last summarization status is stale. No active job is running.")

    cols = st.columns(4)
    cols[0].metric("Progress", f"{completed}/{total}")
    cols[1].metric("Current", current or "-")
    cols[2].metric("Skipped", skipped)
    cols[3].metric("Elapsed", elapsed)
    if failed:
        st.caption(f"Failed chapters: {failed}")
    if status.get("last_saved_summary"):
        st.caption(f"Last saved: {Path(status['last_saved_summary']).name}")


def local_model_config() -> dict[str, Any]:
    model_repo = st.sidebar.text_input(
        "Model repo",
        value="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    )
    model_file = st.sidebar.text_input("Model file", value="*Q4_K_M.gguf")
    context_size = st.sidebar.number_input("Context size", min_value=512, max_value=32768, value=4096, step=512)
    gpu_layers = st.sidebar.number_input("GPU layers", min_value=0, max_value=100, value=20, step=1)
    temperature = st.sidebar.slider(
        "Generation randomness",
        min_value=0.0,
        max_value=1.5,
        value=0.2,
        step=0.05,
        help="Lower values make local model answers and summaries more predictable.",
    )
    return {
        "model_repo": model_repo,
        "model_file": model_file,
        "context_size": int(context_size),
        "gpu_layers": int(gpu_layers),
        "temperature": float(temperature),
    }


def build_summarizer(config: dict[str, Any]):
    return LlamaCppSummarizer(
        model_repo=config["model_repo"],
        model_file=config["model_file"],
        context_size=config["context_size"],
        gpu_layers=config["gpu_layers"],
        temperature=config["temperature"],
    )


def build_story_answerer(config: dict[str, Any]):
    if not config:
        return FakeStoryAnswerer()

    return LlamaCppStoryAnswerer(
        model_repo=config["model_repo"],
        model_file=config["model_file"],
        context_size=config["context_size"],
        gpu_layers=config["gpu_layers"],
        temperature=config["temperature"],
    )


st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
        color: #111827;
    }
    .block-container {
        padding-top: 1.4rem;
        max-width: 1220px;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #d9e2ec;
    }
    [data-testid="stSidebar"] * {
        color: #111827;
    }
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color: #111827;
    }
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff;
        color: #111827;
        border-color: #cbd5e1;
    }
    .stTextArea textarea:disabled {
        color: #111827;
        -webkit-text-fill-color: #111827;
        background: #f8fafc;
    }
    div[data-testid="stTabs"] button p {
        color: #111827;
    }
    div[data-testid="stAlert"] * {
        color: #111827;
    }
    button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }
    button[kind="primary"] p {
        color: #ffffff;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .hero {
        border-bottom: 1px solid #d9e2ec;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }
    .status-card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    .summary-panel {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>LoreLedger</h1>
      <div>Scrape RoyalRoad chapters, summarize selected chapters, and browse character memory.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Library")
    output_root = Path(st.text_input("Output folder", value=str(OUTPUT_ROOT)))
    st.divider()
    st.header("Summaries")
    summarization_enabled = st.checkbox("Enable summarization", value=False)
    force_summary = False
    summarizer_config: dict[str, Any] = {}
    if summarization_enabled:
        force_summary = st.checkbox("Regenerate existing summary", value=False)
        summarizer_config = local_model_config()
        if st.button("Unload local model", use_container_width=True):
            unload_local_models()
            st.success("Requested local model unload.")
    else:
        st.caption("Scrape and read chapters without generating summaries.")

tabs = st.tabs(["Scrape", "Novel", "Character Memory", "Ask Story"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Add chapters")
        title = st.text_input("Novel title", placeholder="Practical Guide To Evil")
        start_url = st.text_input("First chapter URL", placeholder="https://www.royalroad.com/fiction/...")
        scrape_until_end = st.toggle(
            "Scrape until no next chapter",
            value=False,
            help="Ignore the chapter limit and keep following RoyalRoad's next chapter link until it stops.",
        )
        max_chapters = st.number_input(
            "Chapter limit",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            disabled=scrape_until_end,
        )
        delay_seconds = st.slider(
            "Scrape delay between chapters",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="Seconds to wait between RoyalRoad chapter requests.",
        )

        if st.button("Scrape", type="primary", use_container_width=True):
            if not title.strip() or not start_url.strip():
                st.error("Enter both a title and a chapter URL.")
            else:
                try:
                    with st.spinner("Scraping chapters..."):
                        saved = scrape_royalroad(
                            title=title.strip(),
                            start_url=start_url.strip(),
                            max_chapters=None if scrape_until_end else int(max_chapters),
                            delay_seconds=float(delay_seconds),
                            output_root=output_root,
                        )
                    st.success(f"Saved {len(saved)} new chapter(s).")
                except Exception as exc:
                    st.error(f"Scrape failed: {exc}")

    with right:
        st.subheader("Scrape status")
        novels = load_novels(output_root)
        st.metric("Novels", len(novels))
        st.metric("Chapters", sum(item["chapter_count"] for item in novels))
        st.metric("Summaries", sum(item["summary_count"] for item in novels))
        if novels:
            latest = novels[-1]
            st.markdown(
                f"""
                <div class="status-card">
                  <strong>{latest["title"]}</strong><br>
                  {latest["chapter_count"]} chapters stored
                </div>
                """,
                unsafe_allow_html=True,
            )

with tabs[1]:
    novels = load_novels(output_root)
    if not novels:
        st.info("No scraped novels found yet.")
    else:
        selected_novel = st.selectbox("Novel", novels, format_func=novel_label)
        base_dir = novel_dir(selected_novel["slug"], output_root)
        ensure_novel_dirs(base_dir)
        chapters = load_chapters(base_dir)
        job_status = get_summarization_status(base_dir)
        job_running = bool(job_status and job_status.get("status") == "running")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Chapters", selected_novel["chapter_count"])
        metric_cols[1].metric("Summaries", selected_novel["summary_count"])
        metric_cols[2].metric("Characters", selected_novel["character_count"])

        if not chapters:
            st.warning("This novel has no chapter files yet.")
        else:
            chapter = st.selectbox("Chapter", chapters, format_func=chapter_label)
            preview, action = st.columns([1.2, 0.8], gap="large")

            with preview:
                st.subheader(chapter["title"])
                st.caption(chapter["url"])
                st.text_area("Chapter preview", value=chapter["text"][:4000], height=340, disabled=True)

            with action:
                st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
                st.subheader("Summarization")
                summary_path = base_dir / "summaries" / f"chapter_{chapter['number']:04d}.json"
                if summary_path.exists():
                    render_summary(read_json(summary_path))
                else:
                    st.warning("No summary yet.")

                if job_status:
                    render_job_status(job_status)

                if not summarization_enabled:
                    st.info("Summarization is disabled. Enable it in the sidebar to generate summaries.")
                elif st.button(
                    "Summarize selected chapter",
                    type="primary",
                    use_container_width=True,
                    disabled=job_running,
                ):
                    try:
                        start_summarization_job(
                            base_dir=base_dir,
                            novel_slug=selected_novel["slug"],
                            model_config=summarizer_config,
                            start_chapter=int(chapter["number"]),
                            end_chapter=int(chapter["number"]),
                            force=force_summary,
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Summary failed: {exc}")

                if summarization_enabled:
                    chapter_numbers = [int(item["number"]) for item in chapters]
                    min_chapter = min(chapter_numbers)
                    max_chapter = max(chapter_numbers)
                    st.divider()
                    st.write("Batch range")
                    range_cols = st.columns(2)
                    with range_cols[0]:
                        start_chapter = st.number_input(
                            "From chapter",
                            min_value=min_chapter,
                            max_value=max_chapter,
                            value=int(chapter["number"]),
                            step=1,
                        )
                    with range_cols[1]:
                        end_chapter = st.number_input(
                            "To chapter",
                            min_value=min_chapter,
                            max_value=max_chapter,
                            value=max_chapter,
                            step=1,
                        )
                    if st.button("Summarize range", use_container_width=True, disabled=job_running):
                        try:
                            start_summarization_job(
                                base_dir=base_dir,
                                novel_slug=selected_novel["slug"],
                                model_config=summarizer_config,
                                start_chapter=int(start_chapter),
                                end_chapter=int(end_chapter),
                                force=force_summary,
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Batch summary failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    novels = load_novels(output_root)
    if not novels:
        st.info("No novels available.")
    else:
        selected_novel = st.selectbox("Novel for memory", novels, format_func=novel_label)
        base_dir = novel_dir(selected_novel["slug"], output_root)
        chapters = load_chapters(base_dir)
        characters = load_characters(base_dir)

        if not characters:
            st.warning("No character memory yet. Summarize chapters first.")
        else:
            character_names = [character["name"] for character in characters]
            selected_character = st.selectbox("Character", character_names)
            max_chapter = max((chapter["number"] for chapter in chapters), default=1)
            chapter_number = st.slider(
                "Memory through chapter",
                min_value=1,
                max_value=max_chapter,
                value=max_chapter,
                help="Limits the character summary to facts known by that chapter.",
            )

            try:
                st.text_area(
                    "Character summary",
                    value=character_summary_until(base_dir, selected_character, int(chapter_number)),
                    height=340,
                )
            except Exception as exc:
                st.error(str(exc))

with tabs[3]:
    novels = load_novels(output_root)
    if not novels:
        st.info("No novels available.")
    else:
        selected_novel = st.selectbox("Novel to ask about", novels, format_func=novel_label)
        base_dir = novel_dir(selected_novel["slug"], output_root)
        ensure_novel_dirs(base_dir)
        index_path = base_dir / "indexes" / "rag.json"

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.subheader("Ask a question")
            question = st.text_input("Question", placeholder="Who is Arn?")
            top_k = st.slider(
                "Retrieved context count",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of matching story snippets passed to the answerer.",
            )

            button_cols = st.columns(2)
            with button_cols[0]:
                if st.button("Build RAG index", use_container_width=True):
                    try:
                        path = build_rag_index(base_dir, force=True)
                        st.success(f"Saved {path.name}.")
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")

            with button_cols[1]:
                ask_clicked = st.button("Ask", type="primary", use_container_width=True)

            if ask_clicked:
                if not question.strip():
                    st.error("Enter a question.")
                else:
                    try:
                        with st.spinner("Retrieving context and answering..."):
                            build_rag_index(base_dir)
                            answerer = build_story_answerer(summarizer_config)
                            try:
                                result = answer_question(
                                    base_dir,
                                    question.strip(),
                                    answerer,
                                    top_k=int(top_k),
                                )
                            finally:
                                close = getattr(answerer, "close", None)
                                if callable(close):
                                    close()
                        st.write(result["answer"])
                    except Exception as exc:
                        st.error(f"Question failed: {exc}")

        with right:
            st.subheader("Retrieved context")
            if index_path.exists():
                st.success("RAG index exists.")
            else:
                st.warning("No RAG index yet.")

            preview_question = question.strip() if question.strip() else "Who is the main character?"
            try:
                contexts = retrieve_context(base_dir, preview_question, top_k=3) if index_path.exists() else []
                for context in contexts:
                    st.caption(f"{context.reference} | {context.source_type} | {context.score:.3f}")
                    st.write(context.text[:450])
            except Exception as exc:
                st.error(str(exc))

active_status = None
for novel in load_novels(output_root):
    candidate = get_summarization_status(novel_dir(novel["slug"], output_root))
    if candidate and candidate.get("status") == "running":
        active_status = candidate
        break

if active_status:
    time.sleep(1)
    st.rerun()
