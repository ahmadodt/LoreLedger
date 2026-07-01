from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from novel_memory.agent import AgentConfig, PlanAndExecuteAgent, ReActAgent, SimpleRAGAgent
from novel_memory.env import load_project_env
from novel_memory.graph import GRAPH_INDEX_PATH, build_graph, query_graph
from novel_memory.io import read_json, write_json
from novel_memory.memory import character_summary_until
from novel_memory.paths import OUTPUT_ROOT, ensure_novel_dirs, novel_dir
from novel_memory.rag import (
    ConversationTurn,
    FakeStoryAnswerer,
    LlamaCppStoryAnswerer,
    build_bm25_index,
    build_embedding_index,
    build_rag_index,
    retrieve_bm25_context,
    retrieve_embedding_context,
    retrieve_hybrid_context,
    retrieve_story_context,
    retrieve_context,
)
from novel_memory.scraper import iter_chapter_files, scrape_royalroad
from novel_memory.summarization_jobs import (
    cancel_summarization_job,
    elapsed_seconds,
    get_summarization_status,
    start_summarization_job,
    unload_local_models,
)
from novel_memory.summarizer import LlamaCppSummarizer


load_project_env()
st.set_page_config(page_title="LoreLedger", page_icon="LL", layout="wide")

UI_PREFERENCES_FILENAME = ".loreledger_ui.json"
ACTIVE_NOVEL_SESSION_KEY = "active_novel_slug"


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


def ui_preferences_path(output_root: Path) -> Path:
    return output_root / UI_PREFERENCES_FILENAME


def load_ui_preferences(output_root: Path) -> dict[str, Any]:
    path = ui_preferences_path(output_root)
    if not path.exists():
        return {}
    try:
        preferences = read_json(path)
    except (OSError, ValueError):
        return {}
    return preferences if isinstance(preferences, dict) else {}


def save_active_novel_slug(output_root: Path, novel_slug: str) -> None:
    preferences = load_ui_preferences(output_root)
    if preferences.get(ACTIVE_NOVEL_SESSION_KEY) == novel_slug:
        return
    preferences[ACTIVE_NOVEL_SESSION_KEY] = novel_slug
    write_json(ui_preferences_path(output_root), preferences)


def active_novel_index(novels: list[dict[str, Any]], output_root: Path) -> int:
    slugs = [str(novel["slug"]) for novel in novels]
    saved_slug = str(load_ui_preferences(output_root).get(ACTIVE_NOVEL_SESSION_KEY) or "")
    session_slug = str(st.session_state.get(ACTIVE_NOVEL_SESSION_KEY) or "")
    for slug in (saved_slug, session_slug):
        if slug in slugs:
            return slugs.index(slug)
    return 0


def render_active_novel_selector(novels: list[dict[str, Any]], output_root: Path) -> dict[str, Any] | None:
    if not novels:
        return None
    selected_novel = st.selectbox(
        "Active novel",
        novels,
        index=active_novel_index(novels, output_root),
        format_func=novel_label,
        key=f"active_novel_selector_{output_root.as_posix()}",
    )
    st.session_state[ACTIVE_NOVEL_SESSION_KEY] = selected_novel["slug"]
    save_active_novel_slug(output_root, str(selected_novel["slug"]))
    return selected_novel


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


def load_graph_character_names(base_dir: Path) -> list[str]:
    graph_path = base_dir / GRAPH_INDEX_PATH
    if not graph_path.exists():
        return []
    graph = read_json(graph_path)
    return sorted(graph.get("characters", {}), key=str.lower)


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
    chapter_summary = summary.get("chapter_summary", {})
    for label, key in [
        ("Situation", "situation"),
        ("Conflict", "conflict"),
        ("Turning point", "turning_point"),
        ("Consequence", "consequence"),
        ("Hook", "hook"),
    ]:
        value = chapter_summary.get(key, "").strip()
        if value:
            st.markdown(f"**{label}:** {value}")
    pov = summary.get("pov_character")
    if pov:
        st.caption(f"POV: {pov}")
    time_skip = summary.get("time_skip")
    if time_skip:
        st.caption(f"Time skip: {time_skip}")
    locations = summary.get("locations", [])
    if locations:
        st.write("**Locations**")
        for loc in locations:
            name = loc.get("name", "")
            desc = loc.get("description", "")
            st.write(f"- **{name}**: {desc}" if desc else f"- {name}")
    if summary.get("events"):
        st.write("Important events")
        for event in summary["events"]:
            participants = ", ".join(event.get("participants", []))
            event_type = event.get("event_type", "other")
            label = f"{event.get('description', '')} [{event_type}]"
            if participants:
                label = f"{label} - {participants}"
            st.write(f"- {label}")
    elif summary.get("important_events"):
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
    elif state == "cancelled":
        st.warning("Summarization stopped.")
    elif state == "failed":
        st.error(f"Summarization failed: {status.get('error')}")
    elif state == "stale":
        st.warning("Last summarization status is stale. No active job is running.")

    top_cols = st.columns(2)
    top_cols[0].metric("Progress", f"{completed}/{total}")
    top_cols[1].metric("Current", current or "-")
    bottom_cols = st.columns(2)
    bottom_cols[0].metric("Skipped", skipped)
    bottom_cols[1].metric("Elapsed", elapsed)
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
    context_size = st.sidebar.number_input("Context size", min_value=512, max_value=32768, value=8192, step=512)
    gpu_layers = st.sidebar.number_input(
        "GPU layers",
        min_value=-1,
        max_value=100,
        value=20,
        step=1,
        help="Use -1 to request that llama.cpp offload all model layers to the GPU.",
    )
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


AGENT_MODES = {
    "Simple RAG": SimpleRAGAgent,
    "ReAct": ReActAgent,
    "Plan and Execute": PlanAndExecuteAgent,
}
ASK_CONVERSATION_HISTORY_KEY = "ask_conversation_history"
MAX_CONVERSATION_TURNS = 5


def get_conversation_history() -> list[ConversationTurn]:
    raw_history = st.session_state.setdefault(ASK_CONVERSATION_HISTORY_KEY, [])
    history = []
    for turn in raw_history[-MAX_CONVERSATION_TURNS:]:
        if isinstance(turn, ConversationTurn):
            history.append(turn)
        elif isinstance(turn, dict):
            history.append(
                ConversationTurn(
                    question=str(turn.get("question", "")),
                    answer=str(turn.get("answer", "")),
                )
            )
    return history


def append_conversation_turn(question: str, answer: str) -> None:
    raw_history = list(st.session_state.setdefault(ASK_CONVERSATION_HISTORY_KEY, []))
    raw_history.append({"question": question, "answer": answer})
    st.session_state[ASK_CONVERSATION_HISTORY_KEY] = raw_history[-MAX_CONVERSATION_TURNS:]


def reset_conversation_history() -> None:
    st.session_state[ASK_CONVERSATION_HISTORY_KEY] = []


def render_conversation_history(history: list[ConversationTurn]) -> None:
    st.write("Conversation")
    if not history:
        st.caption("No conversation yet.")
        return
    for turn in history:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.write(turn.answer)


def render_agent_steps(steps: list[Any]) -> str:
    if not steps:
        return "_No agent steps yet._"
    lines = []
    for index, step in enumerate(steps, start=1):
        label = step.kind.title()
        query = f" | Query: `{step.query}`" if step.query else ""
        lines.append(f"**{index}. {label}**{query}\n\n{step.content}")
    return "\n\n".join(lines)


def render_relationship_edges(edges: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        grouped.setdefault(str(edge.get("relation", "RELATED")), []).append(edge)

    for relation in sorted(grouped):
        st.write(f"**{relation}**")
        for edge in sorted(grouped[relation], key=lambda item: int(item.get("chapter", 0))):
            evidence = str(edge.get("evidence") or edge.get("description") or "").strip()
            st.write(
                f"Chapter {edge.get('chapter')}: {edge.get('from')} {edge.get('relation')} {edge.get('to')} - \"{evidence}\""
            )


st.markdown(
    """
    <style>
    :root {
        --ll-bg: #F5F3EE;
        --ll-sidebar: #EAE7E0;
        --ll-text: #1C1C1C;
        --ll-muted: #5F5A52;
        --ll-accent: #2A6B6B;
        --ll-accent-hover: #235B5B;
        --ll-border: #C8C4BC;
        --ll-panel: #FFFFFF;
        --ll-shadow: 0 8px 22px rgba(42, 35, 25, 0.08);
        --ll-success-bg: #E8F5E9;
        --ll-success-text: #1B5E20;
        --ll-error-bg: #FDECEA;
        --ll-error-text: #8A1C1C;
        --ll-info-bg: #E6F2F2;
        --ll-info-text: #1F5B5B;
        --ll-warning-bg: #FFF4D8;
        --ll-warning-text: #6A4A00;
    }

    .stApp {
        background: var(--ll-bg);
        color: var(--ll-text);
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1220px;
    }

    [data-testid="stSidebar"] {
        background: var(--ll-sidebar);
        border-right: 1px solid rgba(200, 196, 188, 0.8);
    }

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: var(--ll-sidebar);
    }

    [data-testid="stSidebar"] * {
        color: var(--ll-text);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox label *,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label * {
        color: var(--ll-text);
        -webkit-text-fill-color: var(--ll-text);
    }

    .stApp,
    .stApp p,
    .stApp div,
    .stApp span,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stMarkdown,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: var(--ll-text);
    }

    .stCaptionContainer,
    .stCaptionContainer p,
    small {
        color: var(--ll-muted);
    }

    a,
    a:visited,
    a:hover {
        color: var(--ll-accent);
    }

    hr {
        border-color: rgba(200, 196, 188, 0.7);
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTimeInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background: var(--ll-panel);
        color: var(--ll-text);
        border: 1px solid var(--ll-border);
        border-radius: 8px;
        box-shadow: none;
    }

    .stNumberInput,
    .stNumberInput > div,
    .stNumberInput [data-baseweb="input"],
    .stNumberInput [data-baseweb="input"] > div,
    .stNumberInput [data-baseweb="input"] input,
    .stNumberInput button,
    .stNumberInput button:hover,
    .stNumberInput button:focus {
        background: var(--ll-panel);
        color: var(--ll-text);
        border-color: var(--ll-border);
    }

    .stNumberInput [data-baseweb="input"] {
        border: 1px solid var(--ll-border);
        border-radius: 8px;
        overflow: hidden;
    }

    .stNumberInput [data-baseweb="input"] > div {
        border: 0;
    }

    .stNumberInput button {
        box-shadow: none;
        border-left: 1px solid var(--ll-border);
        border-radius: 0;
    }

    .stNumberInput button *,
    .stNumberInput svg {
        color: var(--ll-text);
        fill: var(--ll-text);
        -webkit-text-fill-color: var(--ll-text);
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--ll-accent);
        box-shadow: 0 0 0 2px rgba(42, 107, 107, 0.16);
    }

    .stTextArea textarea:disabled {
        color: var(--ll-text);
        -webkit-text-fill-color: var(--ll-text);
        background: #FBFAF7;
        border-color: var(--ll-border);
    }

    input::placeholder,
    textarea::placeholder {
        color: #7A746B;
        opacity: 1;
    }

    div[data-testid="stTabs"] button p {
        color: var(--ll-text);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom-color: var(--ll-accent);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--ll-accent);
        font-weight: 700;
    }

    div[data-testid="stAlert"] * {
        color: inherit;
    }

    div[data-testid="stAlert"] {
        border: 0;
        border-radius: 8px;
        box-shadow: var(--ll-shadow);
    }

    div[data-testid="stAlert"][kind="success"],
    div[data-testid="stAlert"]:has(svg[data-testid="stIconMaterial"][aria-label="check_circle"]) {
        background: var(--ll-success-bg);
        color: var(--ll-success-text);
    }

    div[data-testid="stAlert"][kind="error"],
    div[data-testid="stAlert"]:has(svg[data-testid="stIconMaterial"][aria-label="error"]) {
        background: var(--ll-error-bg);
        color: var(--ll-error-text);
    }

    div[data-testid="stAlert"][kind="info"],
    div[data-testid="stAlert"]:has(svg[data-testid="stIconMaterial"][aria-label="info"]) {
        background: var(--ll-info-bg);
        color: var(--ll-info-text);
    }

    div[data-testid="stAlert"][kind="warning"],
    div[data-testid="stAlert"]:has(svg[data-testid="stIconMaterial"][aria-label="warning"]) {
        background: var(--ll-warning-bg);
        color: var(--ll-warning-text);
    }

    .stButton > button,
    .stDownloadButton > button,
    button[kind],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        background: var(--ll-accent);
        border: 1px solid var(--ll-accent);
        border-radius: 8px;
        color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(42, 107, 107, 0.18);
    }

    .stButton > button *,
    .stDownloadButton > button *,
    button[kind] *,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="baseButton-primary"] * {
        color: #FFFFFF;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[kind]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: var(--ll-accent-hover);
        border-color: var(--ll-accent-hover);
        color: #FFFFFF;
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus,
    button[kind]:focus,
    button[data-testid="baseButton-secondary"]:focus,
    button[data-testid="baseButton-primary"]:focus {
        box-shadow: 0 0 0 3px rgba(42, 107, 107, 0.22);
        color: #FFFFFF;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled,
    button[kind]:disabled {
        background: #9BA8A4;
        border-color: #9BA8A4;
        color: #F7F7F5;
        box-shadow: none;
    }

    .stCheckbox label,
    .stRadio label,
    .stToggle label {
        color: var(--ll-text);
        -webkit-text-fill-color: var(--ll-text);
    }

    .stCheckbox label *,
    .stRadio label *,
    .stToggle label *,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label * {
        color: var(--ll-text);
        -webkit-text-fill-color: var(--ll-text);
    }

    .stCheckbox input:checked + div,
    .stRadio input:checked + div,
    .stToggle input:checked + div {
        border-color: var(--ll-accent);
        background: var(--ll-accent);
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--ll-accent);
        border-color: var(--ll-accent);
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background-color: var(--ll-accent);
    }

    div[data-testid="stProgress"] > div {
        background: #DAD6CE;
    }

    div[data-testid="stProgress"] > div > div {
        background: var(--ll-accent);
    }

    button[kind="primary"] {
        background: var(--ll-accent);
        border-color: var(--ll-accent);
        color: #FFFFFF;
    }

    button[kind="primary"] p {
        color: #FFFFFF;
    }

    div[data-testid="stMetric"] {
        background: var(--ll-panel);
        border: 0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: var(--ll-shadow);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ll-text);
    }

    .hero {
        border-bottom: 1px solid rgba(200, 196, 188, 0.7);
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
    }

    .hero h1 {
        font-size: 2.1rem;
        line-height: 1.05;
        margin-bottom: 0.25rem;
        color: var(--ll-text);
    }

    .hero div {
        color: var(--ll-muted);
    }

    .status-card {
        background: var(--ll-panel);
        border: 0;
        border-left: 5px solid var(--ll-accent);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: var(--ll-shadow);
    }

    .stChatMessage {
        background: var(--ll-panel);
        border-radius: 8px;
        box-shadow: var(--ll-shadow);
    }

    div[data-testid="stExpander"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: var(--ll-panel);
        border-radius: 8px;
        box-shadow: var(--ll-shadow);
    }

    code {
        background: #EEEAE2;
        color: var(--ll-text);
        border-radius: 4px;
    }

    ::selection {
        background: rgba(42, 107, 107, 0.22);
        color: var(--ll-text);
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
    sidebar_novels = load_novels(output_root)
    active_novel = render_active_novel_selector(sidebar_novels, output_root)
    st.divider()
    st.header("Summaries")
    summarization_enabled = st.checkbox("Enable summarization", value=False)
    force_summary = False
    summarizer_config: dict[str, Any] = {}
    if summarization_enabled:
        force_summary = st.checkbox("Regenerate existing summary", value=False)
        max_retry_attempts = st.number_input(
            "Max retry attempts",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            key="max_retry_attempts",
        )
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
    if active_novel is None:
        st.info("No scraped novels found yet.")
    else:
        selected_novel = active_novel
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
                st.subheader("Summarization")
                summary_path = base_dir / "summaries" / f"chapter_{chapter['number']:04d}.json"
                if summary_path.exists():
                    render_summary(read_json(summary_path))
                else:
                    st.warning("No summary yet.")

                if job_status:
                    render_job_status(job_status)
                    if job_running and st.button("Stop summarization", use_container_width=True):
                        try:
                            cancel_summarization_job(base_dir)
                            st.warning("Stop requested. The current chapter will finish before the model unloads.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Stop failed: {exc}")

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
                            max_attempts=int(max_retry_attempts),
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
                                max_attempts=int(max_retry_attempts),
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Batch summary failed: {exc}")

        st.divider()
        st.subheader("Relationship Viewer")
        graph_path = base_dir / GRAPH_INDEX_PATH
        if not graph_path.exists():
            st.info("No relationship graph yet.")
            if st.button("Build relationship graph", use_container_width=True):
                try:
                    path = build_graph(base_dir)
                    st.success(f"Saved {path.name}.")
                except Exception as exc:
                    st.error(f"Graph build failed: {exc}")
        else:
            graph_names = load_graph_character_names(base_dir)
            if not graph_names:
                st.info("No relationship nodes found.")
            else:
                selected_graph_name = st.selectbox("Character or faction", graph_names)
                render_relationship_edges(query_graph(base_dir, selected_graph_name))

with tabs[2]:
    if active_novel is None:
        st.info("No novels available.")
    else:
        selected_novel = active_novel
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
    if active_novel is None:
        st.info("No novels available.")
    else:
        selected_novel = active_novel
        base_dir = novel_dir(selected_novel["slug"], output_root)
        ensure_novel_dirs(base_dir)

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.subheader("Ask a question")
            conversation_enabled = st.checkbox("Enable Conversation History", value=True)
            if st.button("New Conversation", use_container_width=True):
                reset_conversation_history()
                st.rerun()
            conversation_history = get_conversation_history() if conversation_enabled else []
            if conversation_enabled:
                render_conversation_history(conversation_history)
            agent_label = st.selectbox("Agent Mode", list(AGENT_MODES))
            retrieval_label = st.radio(
                "Retrieval mode",
                ["TF-IDF", "BM25", "Semantic (Embeddings)", "Hybrid"],
                horizontal=True,
            )
            retrieval_modes = {
                "TF-IDF": ("tfidf", "rag.json"),
                "BM25": ("bm25", "bm25.json"),
                "Semantic (Embeddings)": ("semantic", "embeddings.json"),
                "Hybrid": ("hybrid", "bm25.json"),
            }
            retrieval_mode, index_filename = retrieval_modes[retrieval_label]
            index_path = base_dir / "indexes" / index_filename
            index_exists = (
                index_path.exists()
                and (
                    retrieval_mode != "hybrid"
                    or (base_dir / "indexes" / "embeddings.json").exists()
                )
            )
            rerank_enabled = st.checkbox("Enable Re-ranking", value=False)
            include_graph = st.checkbox("Include Relationship Graph", value=False)
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
                        if retrieval_mode == "semantic":
                            path = build_embedding_index(base_dir, force=True)
                        elif retrieval_mode == "hybrid":
                            build_bm25_index(base_dir, force=True)
                            path = build_embedding_index(base_dir, force=True)
                        elif retrieval_mode == "bm25":
                            path = build_bm25_index(base_dir, force=True)
                        else:
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
                        agent_steps = []
                        reasoning_placeholder = st.empty()
                        with st.spinner("Retrieving context and answering..."):
                            if retrieval_mode == "semantic":
                                build_embedding_index(base_dir)
                            elif retrieval_mode == "hybrid":
                                build_bm25_index(base_dir)
                                build_embedding_index(base_dir)
                            elif retrieval_mode == "bm25":
                                build_bm25_index(base_dir)
                            else:
                                build_rag_index(base_dir)
                            answerer = build_story_answerer(summarizer_config)
                            try:
                                agent = AGENT_MODES[agent_label](
                                    AgentConfig(
                                        retrieval_mode=retrieval_mode,
                                        rerank=rerank_enabled,
                                        top_k=10 if rerank_enabled else int(top_k),
                                        include_graph=include_graph,
                                    )
                                )
                                result = agent.ask(
                                    base_dir,
                                    question.strip(),
                                    answerer,
                                    on_step=lambda step: (
                                        agent_steps.append(step),
                                        reasoning_placeholder.markdown(render_agent_steps(agent_steps)),
                                    ),
                                    conversation_history=conversation_history if conversation_enabled else None,
                                )
                            finally:
                                close = getattr(answerer, "close", None)
                                if callable(close):
                                    close()
                        reasoning_placeholder.markdown(render_agent_steps(result.steps))
                        st.caption(
                            f"Agent: {agent_label} | Retrieval: {retrieval_label} | Re-ranking: {'on' if rerank_enabled else 'off'} | Graph: {'on' if include_graph else 'off'}"
                        )
                        st.write(result.answer)
                        if conversation_enabled:
                            append_conversation_turn(question.strip(), result.answer)
                    except Exception as exc:
                        st.error(f"Question failed: {exc}")

        with right:
            st.subheader("Retrieved context")
            if index_exists:
                st.success("RAG index exists.")
            else:
                st.warning("No RAG index yet.")

            preview_question = question.strip() if question.strip() else "Who is the main character?"
            try:
                if not index_exists:
                    contexts = []
                elif rerank_enabled:
                    contexts = retrieve_story_context(
                        base_dir,
                        preview_question,
                        top_k=10,
                        retrieval_mode=retrieval_mode,
                        rerank=True,
                    )
                elif retrieval_mode == "semantic":
                    contexts = retrieve_embedding_context(base_dir, preview_question, top_k=3)
                elif retrieval_mode == "hybrid":
                    contexts = retrieve_hybrid_context(base_dir, preview_question, top_k=3)
                elif retrieval_mode == "bm25":
                    contexts = retrieve_bm25_context(base_dir, preview_question, top_k=3)
                else:
                    contexts = retrieve_context(base_dir, preview_question, top_k=3)
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
