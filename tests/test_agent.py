from pathlib import Path

from novel_memory.agent import AgentConfig, FakeAgent, FakePlanAndExecuteAgent, PlanAndExecuteAgent, ReActAgent, SimpleRAGAgent
from novel_memory.io import write_json
from novel_memory.paths import ensure_novel_dirs
from novel_memory.rag import ConversationTurn, RetrievedContext


def test_simple_rag_agent_answers_with_citations(monkeypatch, tmp_path: Path):
    context = _context("chapter:0001:001", 1, "Arn survives the arena.")
    calls = []

    def fake_retrieve(base_dir, question, top_k=5, retrieval_mode="tfidf", rerank=False):
        calls.append(
            {
                "base_dir": base_dir,
                "question": question,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "rerank": rerank,
            }
        )
        return [context]

    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", fake_retrieve)
    agent = SimpleRAGAgent(AgentConfig(retrieval_mode="bm25", rerank=True, top_k=7))

    result = agent.ask(tmp_path, "Who is Arn?", FakeAnswerer("Arn is a survivor."))

    assert result.answer.endswith("- Chapter 1 - The Arena")
    assert result.references == ["Chapter 1 - The Arena"]
    assert calls == [
        {
            "base_dir": tmp_path,
            "question": "Who is Arn?",
            "top_k": 7,
            "retrieval_mode": "bm25",
            "rerank": True,
        }
    ]


def test_simple_rag_agent_passes_conversation_history(monkeypatch, tmp_path: Path):
    context = _context("chapter:0001:001", 1, "Arn survives the arena.")
    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", lambda *_args, **_kwargs: [context])
    answerer = FakeAnswerer("Arn is a survivor.")
    history = [ConversationTurn(question="Who is Mira?", answer="Mira is a healer.")]

    SimpleRAGAgent(AgentConfig()).ask(tmp_path, "What about Arn?", answerer, conversation_history=history)

    assert answerer.conversation_history == history


def test_react_agent_retrieves_then_answers(monkeypatch, tmp_path: Path):
    calls = []

    def fake_retrieve(_base_dir, question, top_k=5, retrieval_mode="tfidf", rerank=False):
        calls.append((question, top_k, retrieval_mode, rerank))
        return [_context("chapter:0002:001", 2, "Mira warns Arn about the patron.")]

    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", fake_retrieve)
    answerer = ScriptedAnswerer(
        decisions=[
            '{"thought": "Find Mira and the patron.", "action": "retrieve", "query": "Mira patron warning"}',
            '{"thought": "The context is enough.", "action": "answer"}',
        ],
        answer="Mira warned Arn about the patron.",
    )
    steps = []
    agent = ReActAgent(AgentConfig(retrieval_mode="semantic", rerank=True, top_k=4))

    result = agent.ask(tmp_path, "What did Mira warn Arn about?", answerer, on_step=steps.append)

    assert result.answer.endswith("- Chapter 2 - The Healer")
    assert result.references == ["Chapter 2 - The Healer"]
    assert calls == [("Mira patron warning", 4, "semantic", True)]
    assert [step.kind for step in result.steps[:4]] == ["think", "act", "observe", "think"]
    assert [step.kind for step in steps] == ["think", "act", "observe", "think"]


def test_react_agent_passes_conversation_history(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "novel_memory.agent.retrieve_story_context",
        lambda *_args, **_kwargs: [_context("chapter:0001:001", 1, "Arn survives the arena.")],
    )
    answerer = ScriptedAnswerer(
        decisions=[
            '{"thought": "Find Arn.", "action": "retrieve", "query": "Arn"}',
            '{"thought": "The context is enough.", "action": "answer"}',
        ],
        answer="Arn is a survivor.",
    )
    history = [ConversationTurn(question="Who is Mira?", answer="Mira is a healer.")]

    ReActAgent(AgentConfig()).ask(tmp_path, "What about Arn?", answerer, conversation_history=history)

    assert answerer.conversation_history == history


def test_react_agent_forces_final_answer_after_three_loops(monkeypatch, tmp_path: Path):
    calls = []

    def fake_retrieve(_base_dir, question, **_kwargs):
        calls.append(question)
        return [_context(f"chapter:000{len(calls)}:001", len(calls), f"Context {len(calls)}")]

    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", fake_retrieve)
    answerer = ScriptedAnswerer(
        decisions=[
            '{"thought": "Search one.", "action": "retrieve", "query": "one"}',
            '{"thought": "Search two.", "action": "retrieve", "query": "two"}',
            '{"thought": "Search three.", "action": "retrieve", "query": "three"}',
        ],
        answer="Forced final answer.",
    )
    agent = ReActAgent(AgentConfig(top_k=2))

    result = agent.ask(tmp_path, "What happened?", answerer)

    assert calls == ["one", "two", "three"]
    assert result.answer.startswith("Forced final answer.")
    assert len([step for step in result.steps if step.kind == "act"]) == 3
    assert result.steps[-1].kind == "final"


def test_react_agent_reports_missing_context_after_empty_loops(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", lambda *_args, **_kwargs: [])
    answerer = ScriptedAnswerer(
        decisions=[
            '{"thought": "Search one.", "action": "retrieve", "query": "one"}',
            '{"thought": "Search two.", "action": "retrieve", "query": "two"}',
            '{"thought": "Search three.", "action": "retrieve", "query": "three"}',
        ],
        answer="Should not be used.",
    )

    result = ReActAgent(AgentConfig()).ask(tmp_path, "Who is Nobody?", answerer)

    assert result.answer == "I do not have enough stored context to answer that."
    assert result.references == []


def test_react_agent_queries_graph_when_name_detected(monkeypatch, tmp_path: Path):
    _write_agent_graph(tmp_path)
    graph_calls = []
    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", lambda *_args, **_kwargs: [])

    def fake_query_graph(_base_dir, name):
        graph_calls.append(name)
        return [_graph_edge()]

    monkeypatch.setattr("novel_memory.agent.query_graph", fake_query_graph)
    answerer = ScriptedAnswerer(
        decisions=[
            '{"thought": "Find Arn relationships.", "action": "retrieve", "query": "Arn"}',
            '{"thought": "Graph context is enough.", "action": "answer"}',
        ],
        answer="Arn saved Mira.",
    )

    result = ReActAgent(AgentConfig(include_graph=True)).ask(tmp_path, "What is Arn's relationship with Mira?", answerer)

    assert set(graph_calls) == {"Arn", "Mira"}
    assert any(context.source_type == "graph" for context in result.contexts)
    assert "Chapter 7 - Graph" in result.answer


def test_plan_and_execute_agent_generates_plan_and_retrieves_each_step(monkeypatch, tmp_path: Path):
    calls = []

    def fake_retrieve(_base_dir, question, top_k=5, retrieval_mode="tfidf", rerank=False):
        calls.append((question, top_k, retrieval_mode, rerank))
        return [
            _context(
                f"chapter:000{len(calls)}:001",
                len(calls),
                f"Context for {question}",
            )
        ]

    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", fake_retrieve)
    answerer = ScriptedAnswerer(
        decisions=['["Find who Arn is", "Find Arn relationship with Mira", "Find how it changed"]'],
        answer="Arn and Mira changed over time.",
    )
    steps = []
    agent = PlanAndExecuteAgent(AgentConfig(retrieval_mode="hybrid", rerank=True, top_k=6))

    result = agent.ask(tmp_path, "How did Arn and Mira change?", answerer, on_step=steps.append)

    assert calls == [
        ("Find who Arn is", 6, "hybrid", True),
        ("Find Arn relationship with Mira", 6, "hybrid", True),
        ("Find how it changed", 6, "hybrid", True),
    ]
    assert result.answer.startswith("Arn and Mira changed over time.")
    assert "Chapter 1 - The Arena" in result.answer
    assert result.references == ["Chapter 1 - The Arena", "Chapter 2 - The Healer", "Chapter 3 - Chapter 3"]
    assert [step.kind for step in steps[:4]] == ["plan", "act", "observe", "act"]


def test_plan_and_execute_agent_passes_conversation_history(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "novel_memory.agent.retrieve_story_context",
        lambda *_args, **_kwargs: [_context("chapter:0001:001", 1, "Arn survives the arena.")],
    )
    answerer = ScriptedAnswerer(decisions=['["Find Arn"]'], answer="Arn is a survivor.")
    history = [ConversationTurn(question="Who is Mira?", answer="Mira is a healer.")]

    PlanAndExecuteAgent(AgentConfig()).ask(tmp_path, "What about Arn?", answerer, conversation_history=history)

    assert answerer.conversation_history == history


def test_plan_and_execute_agent_includes_graph_context(monkeypatch, tmp_path: Path):
    _write_agent_graph(tmp_path)
    graph_calls = []
    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", lambda *_args, **_kwargs: [_context("chapter:0001:001", 1, "RAG context.")])

    def fake_query_graph(_base_dir, name):
        graph_calls.append(name)
        return [_graph_edge()]

    monkeypatch.setattr("novel_memory.agent.query_graph", fake_query_graph)
    answerer = ScriptedAnswerer(decisions=['["Find Arn and Mira relationship"]'], answer="Combined answer.")

    result = PlanAndExecuteAgent(AgentConfig(include_graph=True)).ask(tmp_path, "How are Arn and Mira connected?", answerer)

    assert set(graph_calls) == {"Arn", "Mira"}
    assert [context.source_type for context in result.contexts] == ["chapter", "graph"]
    assert "Chapter 7 - Graph" in result.answer


def test_simple_rag_agent_does_not_query_graph(monkeypatch, tmp_path: Path):
    _write_agent_graph(tmp_path)
    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", lambda *_args, **_kwargs: [_context("chapter:0001:001", 1, "RAG context.")])
    monkeypatch.setattr("novel_memory.agent.query_graph", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("graph should not be queried")))

    result = SimpleRAGAgent(AgentConfig(include_graph=True)).ask(tmp_path, "How are Arn and Mira connected?", FakeAnswerer("RAG answer."))

    assert [context.source_type for context in result.contexts] == ["chapter"]


def test_plan_and_execute_agent_truncates_plan_to_five_steps(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(
        "novel_memory.agent.retrieve_story_context",
        lambda _base_dir, question, **_kwargs: calls.append(question) or [_context(f"chapter:{len(calls):04d}:001", len(calls), question)],
    )
    answerer = ScriptedAnswerer(
        decisions=['["one", "two", "three", "four", "five", "six"]'],
        answer="Answer.",
    )

    result = PlanAndExecuteAgent(AgentConfig()).ask(tmp_path, "Question?", answerer)

    assert calls == ["one", "two", "three", "four", "five"]
    assert "six" not in result.steps[0].content


def test_plan_and_execute_agent_deduplicates_combined_contexts(monkeypatch, tmp_path: Path):
    shared = _context("chapter:0001:001", 1, "Shared context.")

    def fake_retrieve(_base_dir, question, **_kwargs):
        if question == "first":
            return [shared, _context("chapter:0002:001", 2, "Second context.")]
        return [shared, _context("chapter:0003:001", 3, "Third context.")]

    monkeypatch.setattr("novel_memory.agent.retrieve_story_context", fake_retrieve)
    answerer = ScriptedAnswerer(decisions=['["first", "second"]'], answer="Combined answer.")

    result = PlanAndExecuteAgent(AgentConfig()).ask(tmp_path, "Question?", answerer)

    assert [context.id for context in result.contexts] == [
        "chapter:0001:001",
        "chapter:0002:001",
        "chapter:0003:001",
    ]
    assert result.answer.endswith("- Chapter 3 - Chapter 3")


def test_fake_agent_returns_predictable_output(tmp_path: Path):
    result = FakeAgent(answer="Predictable.", references=["Chapter 1 - Test"]).ask(
        tmp_path,
        "Question?",
        FakeAnswerer("unused"),
    )

    assert result.answer == "Predictable."
    assert result.references == ["Chapter 1 - Test"]
    assert result.steps[0].kind == "final"


def test_fake_plan_and_execute_agent_returns_predictable_output(tmp_path: Path):
    steps = []
    result = FakePlanAndExecuteAgent(
        plan=["Find one", "Find two"],
        answer="Predictable plan answer.",
        references=["Chapter 1 - Test"],
    ).ask(tmp_path, "Question?", FakeAnswerer("unused"), on_step=steps.append)

    assert result.answer == "Predictable plan answer."
    assert result.references == ["Chapter 1 - Test"]
    assert [step.kind for step in result.steps] == ["plan", "act", "observe", "act", "observe", "final"]
    assert [step.kind for step in steps] == ["plan", "act", "observe", "act", "observe", "final"]


class FakeAnswerer:
    def __init__(self, answer: str):
        self.answer = answer
        self.conversation_history = None

    def answer_question(self, _question, _contexts, conversation_history=None):
        self.conversation_history = conversation_history
        return self.answer


class ScriptedAnswerer(FakeAnswerer):
    def __init__(self, decisions: list[str], answer: str):
        super().__init__(answer)
        self.decisions = decisions
        self.prompt_count = 0

    def complete_prompt(self, _prompt, **_kwargs):
        decision = self.decisions[min(self.prompt_count, len(self.decisions) - 1)]
        self.prompt_count += 1
        return decision


def _context(id: str, chapter_number: int, text: str) -> RetrievedContext:
    title = {1: "The Arena", 2: "The Healer"}.get(chapter_number, f"Chapter {chapter_number}")
    return RetrievedContext(
        id=id,
        source_type="chapter",
        chapter_number=chapter_number,
        chapter_title=title,
        text=text,
        score=1.0,
    )


def _write_agent_graph(base_dir: Path) -> None:
    ensure_novel_dirs(base_dir)
    write_json(
        base_dir / "indexes" / "graph.json",
        {
            "characters": {
                "Arn": {"type": "character", "first_seen": 1, "aliases": []},
                "Mira": {"type": "character", "first_seen": 1, "aliases": []},
            },
            "edges": [_graph_edge()],
        },
    )


def _graph_edge() -> dict:
    return {
        "from": "Arn",
        "to": "Mira",
        "relation": "SAVED",
        "chapter": 7,
        "description": "Arn saved Mira.",
        "evidence": "He stepped in front of her.",
    }
