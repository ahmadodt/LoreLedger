from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .graph import GRAPH_INDEX_PATH, query_graph
from .io import read_json
from .rag import (
    RetrievedContext,
    StoryAnswerer,
    _ensure_references,
    _unique_references,
    retrieve_story_context,
)


MAX_REACT_LOOPS = 3
MAX_PLAN_STEPS = 5
INSUFFICIENT_CONTEXT_ANSWER = "I do not have enough stored context to answer that."


@dataclass(frozen=True)
class AgentConfig:
    retrieval_mode: str = "tfidf"
    rerank: bool = False
    top_k: int = 5
    include_graph: bool = False


@dataclass(frozen=True)
class AgentStep:
    kind: str
    content: str
    query: str | None = None
    contexts: list[RetrievedContext] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    references: list[str]
    contexts: list[RetrievedContext]
    steps: list[AgentStep]


class Agent(Protocol):
    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        ...


@dataclass
class SimpleRAGAgent:
    config: AgentConfig

    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        contexts = retrieve_story_context(
            base_dir,
            question,
            top_k=self.config.top_k,
            retrieval_mode=self.config.retrieval_mode,
            rerank=self.config.rerank,
        )
        steps = [
            AgentStep(
                kind="observe",
                content=f"Retrieved {len(contexts)} context chunk(s).",
                query=question,
                contexts=contexts,
            )
        ]
        _emit_step(steps[-1], on_step)
        return _answer_from_contexts(question, contexts, answerer, steps)


@dataclass
class ReActAgent:
    config: AgentConfig
    max_loops: int = MAX_REACT_LOOPS

    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        steps: list[AgentStep] = []
        accumulated_contexts: list[RetrievedContext] = []
        seen_context_ids: set[str] = set()
        next_query = question

        for loop_index in range(1, self.max_loops + 1):
            decision = _react_decision(answerer, question, steps, next_query, loop_index)
            thought = str(decision.get("thought") or "I need to search the stored story context.").strip()
            action = str(decision.get("action") or "retrieve").strip().lower()
            refined_query = str(decision.get("query") or next_query or question).strip() or question

            think_step = AgentStep(kind="think", content=thought, query=refined_query)
            steps.append(think_step)
            _emit_step(think_step, on_step)

            if action == "answer":
                break

            act_step = AgentStep(kind="act", content=f"Retrieve context for: {refined_query}", query=refined_query)
            steps.append(act_step)
            _emit_step(act_step, on_step)

            contexts = retrieve_story_context(
                base_dir,
                refined_query,
                top_k=self.config.top_k,
                retrieval_mode=self.config.retrieval_mode,
                rerank=self.config.rerank,
            )
            _extend_unique_contexts(accumulated_contexts, seen_context_ids, contexts)

            observe_step = AgentStep(
                kind="observe",
                content=_observation_text(contexts),
                query=refined_query,
                contexts=contexts,
            )
            steps.append(observe_step)
            _emit_step(observe_step, on_step)
            if self.config.include_graph:
                graph_contexts = _query_graph_contexts(base_dir, question, refined_query, steps, on_step)
                _extend_unique_contexts(accumulated_contexts, seen_context_ids, graph_contexts)
            next_query = refined_query

        return _answer_from_contexts(question, accumulated_contexts, answerer, steps)


@dataclass
class PlanAndExecuteAgent:
    config: AgentConfig
    max_steps: int = MAX_PLAN_STEPS

    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        steps: list[AgentStep] = []
        plan = _build_plan(answerer, question, self.max_steps)
        plan_step = AgentStep(kind="plan", content="\n".join(f"{index}. {item}" for index, item in enumerate(plan, start=1)))
        steps.append(plan_step)
        _emit_step(plan_step, on_step)

        accumulated_contexts: list[RetrievedContext] = []
        seen_context_ids: set[str] = set()
        for item in plan:
            act_step = AgentStep(kind="act", content=f"Retrieve context for: {item}", query=item)
            steps.append(act_step)
            _emit_step(act_step, on_step)

            contexts = retrieve_story_context(
                base_dir,
                item,
                top_k=self.config.top_k,
                retrieval_mode=self.config.retrieval_mode,
                rerank=self.config.rerank,
            )
            _extend_unique_contexts(accumulated_contexts, seen_context_ids, contexts)

            observe_step = AgentStep(
                kind="observe",
                content=_observation_text(contexts),
                query=item,
                contexts=contexts,
            )
            steps.append(observe_step)
            _emit_step(observe_step, on_step)
            if self.config.include_graph:
                graph_contexts = _query_graph_contexts(base_dir, question, item, steps, on_step)
                _extend_unique_contexts(accumulated_contexts, seen_context_ids, graph_contexts)

        return _answer_from_contexts(question, accumulated_contexts, answerer, steps)


@dataclass
class FakeAgent:
    answer: str = "Fake agent answer."
    references: list[str] = field(default_factory=lambda: ["Chapter 1 - Fake"])

    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        step = AgentStep(kind="final", content=f"Fake answer for: {question}")
        _emit_step(step, on_step)
        return AgentResult(answer=self.answer, references=list(self.references), contexts=[], steps=[step])


@dataclass
class FakePlanAndExecuteAgent:
    plan: list[str] = field(default_factory=lambda: ["Find fake context"])
    answer: str = "Fake plan and execute answer."
    references: list[str] = field(default_factory=lambda: ["Chapter 1 - Fake"])

    def ask(
        self,
        base_dir: Path,
        question: str,
        answerer: StoryAnswerer,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        steps = [AgentStep(kind="plan", content="\n".join(self.plan))]
        _emit_step(steps[-1], on_step)
        for item in self.plan:
            steps.append(AgentStep(kind="act", content=f"Retrieve context for: {item}", query=item))
            _emit_step(steps[-1], on_step)
            steps.append(AgentStep(kind="observe", content="Fake observation.", query=item))
            _emit_step(steps[-1], on_step)
        steps.append(AgentStep(kind="final", content=f"Fake answer for: {question}"))
        _emit_step(steps[-1], on_step)
        return AgentResult(answer=self.answer, references=list(self.references), contexts=[], steps=steps)


def _answer_from_contexts(
    question: str,
    contexts: list[RetrievedContext],
    answerer: StoryAnswerer,
    steps: list[AgentStep],
) -> AgentResult:
    if not contexts:
        final_step = AgentStep(kind="final", content=INSUFFICIENT_CONTEXT_ANSWER)
        steps.append(final_step)
        return AgentResult(answer=INSUFFICIENT_CONTEXT_ANSWER, references=[], contexts=[], steps=steps)

    answer = answerer.answer_question(question, contexts).strip()
    references = _unique_references(contexts)
    answer = _ensure_references(answer, references)
    steps.append(AgentStep(kind="final", content=answer, contexts=contexts))
    return AgentResult(answer=answer, references=references, contexts=contexts, steps=steps)


def _query_graph_contexts(
    base_dir: Path,
    question: str,
    query: str,
    steps: list[AgentStep],
    on_step: Callable[[AgentStep], None] | None,
) -> list[RetrievedContext]:
    contexts: list[RetrievedContext] = []
    for name in _candidate_graph_names(base_dir, f"{question} {query}"):
        act_step = AgentStep(kind="act", content=f"Query relationship graph for: {name}", query=name)
        steps.append(act_step)
        _emit_step(act_step, on_step)

        graph_contexts = _graph_contexts(base_dir, name)
        contexts.extend(graph_contexts)
        observe_step = AgentStep(
            kind="observe",
            content=f"Found {len(graph_contexts)} relationship edge(s).",
            query=name,
            contexts=graph_contexts,
        )
        steps.append(observe_step)
        _emit_step(observe_step, on_step)
    return contexts


def _candidate_graph_names(base_dir: Path, text: str) -> list[str]:
    graph_path = base_dir / GRAPH_INDEX_PATH
    if not graph_path.exists() or not text.strip():
        return []
    graph = read_json(graph_path)
    matched = []
    for name in sorted(graph.get("characters", {}), key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(str(name))}(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(str(name))
    return matched


def _graph_contexts(base_dir: Path, name: str) -> list[RetrievedContext]:
    contexts = []
    seen = set()
    for edge in query_graph(base_dir, name):
        context = _graph_edge_context(edge)
        if context.id in seen:
            continue
        seen.add(context.id)
        contexts.append(context)
    return contexts


def _graph_edge_context(edge: dict[str, Any]) -> RetrievedContext:
    chapter = int(edge.get("chapter", 0))
    source = str(edge.get("from", "")).strip()
    target = str(edge.get("to", "")).strip()
    relation = str(edge.get("relation", "RELATED")).strip()
    description = str(edge.get("description", "")).strip()
    evidence = str(edge.get("evidence", "")).strip()
    text = f"{source} {relation} {target}."
    if description:
        text = f"{text} {description}"
    if evidence:
        text = f"{text} Evidence: {evidence}"
    return RetrievedContext(
        id=f"graph:{chapter}:{source}:{relation}:{target}:{evidence}",
        source_type="graph",
        chapter_number=chapter,
        chapter_title="Graph",
        text=text,
        score=1.0,
    )


def _extend_unique_contexts(
    accumulated_contexts: list[RetrievedContext],
    seen_context_ids: set[str],
    contexts: list[RetrievedContext],
) -> None:
    for context in contexts:
        if context.id in seen_context_ids:
            continue
        seen_context_ids.add(context.id)
        accumulated_contexts.append(context)


def _build_plan(answerer: StoryAnswerer, question: str, max_steps: int = MAX_PLAN_STEPS) -> list[str]:
    complete = getattr(answerer, "complete_prompt", None)
    if not callable(complete):
        return [question]

    prompt = _build_plan_prompt(question, max_steps)
    try:
        text = complete(prompt, stop=None, max_tokens=320)
    except Exception:
        return [question]

    plan = _parse_plan(text)
    if not plan:
        return [question]
    return plan[:max_steps]


def _parse_plan(text: str) -> list[str]:
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, list):
        return _clean_plan_items(parsed)
    if isinstance(parsed, dict):
        for key in ("plan", "steps", "queries"):
            if isinstance(parsed.get(key), list):
                return _clean_plan_items(parsed[key])

    line_items = []
    for line in text.splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if item and item != line.strip():
            line_items.append(item)
    return _clean_plan_items(line_items)


def _clean_plan_items(items: list[Any]) -> list[str]:
    cleaned = []
    for item in items:
        value = str(item).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _build_plan_prompt(question: str, max_steps: int) -> str:
    return f"""You are LoreLedger's planning agent.

Create a step-by-step retrieval plan for answering the user's story question.
Return only a strict JSON array of strings. Do not answer the question.
Use at most {max_steps} steps.

Question: {question}
"""


def _react_decision(
    answerer: StoryAnswerer,
    question: str,
    steps: list[AgentStep],
    next_query: str,
    loop_index: int,
) -> dict[str, Any]:
    prompt = _build_react_prompt(question, steps, next_query, loop_index)
    complete = getattr(answerer, "complete_prompt", None)
    if not callable(complete):
        return {"thought": "Search the stored story context.", "action": "retrieve", "query": next_query}

    try:
        text = complete(prompt, stop=None, max_tokens=220)
        return _parse_react_decision(text)
    except Exception:
        return {"thought": "Search the stored story context.", "action": "retrieve", "query": next_query}


def _parse_react_decision(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"thought": "Search the stored story context.", "action": "retrieve"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"thought": "Search the stored story context.", "action": "retrieve"}

    if not isinstance(parsed, dict):
        return {"thought": "Search the stored story context.", "action": "retrieve"}
    action = str(parsed.get("action", "retrieve")).lower()
    if action not in {"retrieve", "answer"}:
        parsed["action"] = "retrieve"
    return parsed


def _build_react_prompt(question: str, steps: list[AgentStep], next_query: str, loop_index: int) -> str:
    trace = "\n".join(
        f"{step.kind.title()}: {step.content}"
        + (f"\nQuery: {step.query}" if step.query else "")
        + (_context_summary(step.contexts) if step.contexts else "")
        for step in steps[-6:]
    )
    return f"""You are LoreLedger's ReAct story QA agent.

Decide the next step for answering the user's question from stored story context.
Return only strict JSON with keys: thought, action, query.
Valid action values are "retrieve" or "answer".
Use "retrieve" when another search is needed. Use "answer" when the observations are enough.

Question: {question}
Loop: {loop_index}/{MAX_REACT_LOOPS}
Suggested next query: {next_query}

Trace:
{trace or "No prior steps."}
"""


def _observation_text(contexts: list[RetrievedContext]) -> str:
    if not contexts:
        return "No context chunks found."
    references = ", ".join(_unique_references(contexts))
    return f"Found {len(contexts)} context chunk(s): {references}"


def _context_summary(contexts: list[RetrievedContext]) -> str:
    lines = []
    for context in contexts[:3]:
        lines.append(f"\n- {context.reference} [{context.source_type}]: {context.text[:220]}")
    return "".join(lines)


def _emit_step(step: AgentStep, on_step: Callable[[AgentStep], None] | None) -> None:
    if on_step is not None:
        on_step(step)
