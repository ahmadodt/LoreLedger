from pathlib import Path

from novel_memory.agent import AgentConfig, FakeAgent, ReActAgent, SimpleRAGAgent
from novel_memory.rag import RetrievedContext


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


def test_fake_agent_returns_predictable_output(tmp_path: Path):
    result = FakeAgent(answer="Predictable.", references=["Chapter 1 - Test"]).ask(
        tmp_path,
        "Question?",
        FakeAnswerer("unused"),
    )

    assert result.answer == "Predictable."
    assert result.references == ["Chapter 1 - Test"]
    assert result.steps[0].kind == "final"


class FakeAnswerer:
    def __init__(self, answer: str):
        self.answer = answer

    def answer_question(self, _question, _contexts):
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
