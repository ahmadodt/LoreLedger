from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from novel_memory.io import read_json
from novel_memory.memory import character_summary_until
from novel_memory.paths import OUTPUT_ROOT, ensure_novel_dirs, novel_dir
from novel_memory.scraper import iter_chapter_files, scrape_royalroad
from novel_memory.summarizer import FakeSummarizer, LlamaCppSummarizer, summarize_chapter


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
        novels.append(
            {
                "slug": path.name,
                "title": metadata.get("title", path.name.replace("_", " ").title()),
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
        chapters.append(
            {
                "number": int(chapter["number"]),
                "title": chapter["title"],
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
        characters.append(
            {
                "name": character.get("name", path.stem),
                "aliases": character.get("aliases", []),
                "timeline": character.get("timeline", []),
            }
        )
    return sorted(characters, key=lambda item: item["name"].lower())


def chapter_label(chapter: dict[str, Any]) -> str:
    return f"{chapter['number']:04d} - {chapter['title']}"


def summary_engine_config() -> dict[str, Any]:
    engine = st.sidebar.radio("Summary engine", ["Demo", "Local GGUF"], horizontal=True)
    if engine == "Demo":
        return {"engine": "Demo"}

    model_repo = st.sidebar.text_input(
        "Model repo",
        value="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    )
    model_file = st.sidebar.text_input("Model file", value="*Q4_K_M.gguf")
    context_size = st.sidebar.number_input("Context size", min_value=512, max_value=32768, value=4096, step=512)
    gpu_layers = st.sidebar.number_input("GPU layers", min_value=0, max_value=100, value=20, step=1)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.05)
    return {
        "engine": "Local GGUF",
        "model_repo": model_repo,
        "model_file": model_file,
        "context_size": int(context_size),
        "gpu_layers": int(gpu_layers),
        "temperature": float(temperature),
    }


def build_summarizer(config: dict[str, Any]):
    if config["engine"] == "Demo":
        return FakeSummarizer()

    return LlamaCppSummarizer(
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
        background: #f6f4ef;
        color: #1f2523;
    }
    .block-container {
        padding-top: 1.4rem;
        max-width: 1220px;
    }
    [data-testid="stSidebar"] {
        background: #26312f;
    }
    [data-testid="stSidebar"] * {
        color: #f7f3e8;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8d1c3;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .hero {
        border-bottom: 1px solid #d8d1c3;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }
    .status-card {
        background: #fffdf8;
        border: 1px solid #d8d1c3;
        border-left: 5px solid #2f6f73;
        border-radius: 8px;
        padding: 0.9rem 1rem;
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
    force_summary = st.checkbox("Regenerate existing summary", value=False)
    summarizer_config = summary_engine_config()

tabs = st.tabs(["Scrape", "Novel", "Character Memory"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Add chapters")
        title = st.text_input("Novel title", placeholder="Practical Guide To Evil")
        start_url = st.text_input("First chapter URL", placeholder="https://www.royalroad.com/fiction/...")
        max_chapters = st.number_input("Chapter limit", min_value=1, max_value=500, value=5, step=1)
        delay_seconds = st.slider("Delay between chapters", min_value=0.0, max_value=5.0, value=1.0, step=0.25)

        if st.button("Scrape", type="primary", use_container_width=True):
            if not title.strip() or not start_url.strip():
                st.error("Enter both a title and a chapter URL.")
            else:
                try:
                    with st.spinner("Scraping chapters..."):
                        saved = scrape_royalroad(
                            title=title.strip(),
                            start_url=start_url.strip(),
                            max_chapters=int(max_chapters),
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
        selected_title = st.selectbox("Novel", [novel["title"] for novel in novels])
        selected_novel = next(novel for novel in novels if novel["title"] == selected_title)
        base_dir = novel_dir(selected_novel["slug"], output_root)
        ensure_novel_dirs(base_dir)
        chapters = load_chapters(base_dir)

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
                st.subheader("Chapter summary")
                summary_path = base_dir / "summaries" / f"chapter_{chapter['number']:04d}.json"
                if summary_path.exists():
                    summary = read_json(summary_path)
                    st.success("Summary exists.")
                    st.write(summary.get("chapter_summary", ""))
                    if summary.get("important_events"):
                        st.write("Important events")
                        for event in summary["important_events"]:
                            st.write(f"- {event}")
                else:
                    st.warning("No summary yet.")

                if st.button("Summarize selected chapter", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Summarizing chapter..."):
                            path = summarize_chapter(
                                base_dir,
                                int(chapter["number"]),
                                build_summarizer(summarizer_config),
                                force=force_summary,
                            )
                        st.success(f"Saved {path.name}.")
                    except Exception as exc:
                        st.error(f"Summary failed: {exc}")

with tabs[2]:
    novels = load_novels(output_root)
    if not novels:
        st.info("No novels available.")
    else:
        selected_title = st.selectbox("Novel for memory", [novel["title"] for novel in novels])
        selected_novel = next(novel for novel in novels if novel["title"] == selected_title)
        base_dir = novel_dir(selected_novel["slug"], output_root)
        chapters = load_chapters(base_dir)
        characters = load_characters(base_dir)

        if not characters:
            st.warning("No character memory yet. Summarize chapters first.")
        else:
            character_names = [character["name"] for character in characters]
            selected_character = st.selectbox("Character", character_names)
            max_chapter = max((chapter["number"] for chapter in chapters), default=1)
            chapter_number = st.slider("Through chapter", min_value=1, max_value=max_chapter, value=max_chapter)

            try:
                st.text_area(
                    "Character summary",
                    value=character_summary_until(base_dir, selected_character, int(chapter_number)),
                    height=340,
                )
            except Exception as exc:
                st.error(str(exc))
