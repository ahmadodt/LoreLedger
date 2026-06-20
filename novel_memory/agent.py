from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .rag import (
    RetrievedContext,
    StoryAnswerer,
    _ensure_references,
    _unique_references,
    retrieve_story_context,
)


MAX_REACT_LOOPS = 3
INSUFFICIENT_CONTEXT_ANSWER = "I do not have enough stored context to answer that."


@dataclass(frozen=True)
class AgentConfig:
    retrieval_mode: str = "tfidf"
    rerank: bool = False
    top_k: int = 5


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
            for context in contexts:
                if context.id in seen_context_ids:
                    continue
                seen_context_ids.add(context.id)
                accumulated_contexts.append(context)

            observe_step = AgentStep(
                kind="observe",
                content=_observation_text(contexts),
                query=refined_query,
                contexts=contexts,
            )
            steps.append(observe_step)
            _emit_step(observe_step, on_step)
            next_query = refined_query

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
