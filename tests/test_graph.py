from pathlib import Path

from novel_memory.graph import build_graph, query_graph
from novel_memory.io import read_json, write_json
from novel_memory.paths import ensure_novel_dirs


def test_build_graph_extracts_relationship_edges_and_nodes(tmp_path: Path):
    _write_graph_fixture(tmp_path)

    path = build_graph(tmp_path)

    graph = read_json(path)
    assert graph["characters"]["Arn"] == {"type": "character", "first_seen": 1, "aliases": []}
    assert graph["characters"]["Mira"] == {"type": "character", "first_seen": 1, "aliases": []}
    assert graph["edges"][0] == {
        "from": "Arn",
        "to": "Mira",
        "relation": "SAVED",
        "chapter": 1,
        "description": "Arn saved Mira from the arena guard.",
        "evidence": "Arn pulled Mira away from the guard.",
    }


def test_build_graph_marks_known_faction_participants(tmp_path: Path):
    _write_graph_fixture(tmp_path)

    graph = read_json(build_graph(tmp_path))

    assert graph["characters"]["Guild"]["type"] == "faction"
    assert graph["characters"]["Mira"]["type"] == "character"
    guild_edges = [edge for edge in graph["edges"] if edge["to"] == "Guild"]
    assert guild_edges[0]["relation"] == "JOINED"


def test_query_graph_matches_character_name_case_insensitively(tmp_path: Path):
    _write_graph_fixture(tmp_path)
    build_graph(tmp_path)

    edges = query_graph(tmp_path, "mira")

    assert len(edges) == 2
    assert {edge["relation"] for edge in edges} == {"SAVED", "JOINED"}


def test_query_graph_returns_empty_for_unknown_character(tmp_path: Path):
    _write_graph_fixture(tmp_path)
    build_graph(tmp_path)

    assert query_graph(tmp_path, "Nobody") == []


def _write_graph_fixture(base_dir: Path) -> None:
    ensure_novel_dirs(base_dir)
    write_json(
        base_dir / "summaries" / "chapter_0001.json",
        {
            "chapter_number": 1,
            "chapter_title": "The Arena",
            "events": [
                {
                    "description": "Arn saved Mira from the arena guard.",
                    "event_type": "relationship",
                    "participants": ["Arn", "Mira"],
                    "evidence": "Arn pulled Mira away from the guard.",
                },
                {
                    "description": "The Guild recruits Mira.",
                    "event_type": "faction",
                    "participants": ["Guild"],
                    "evidence": "The Guild offers Mira a place.",
                },
            ],
        },
    )
    write_json(
        base_dir / "summaries" / "chapter_0002.json",
        {
            "chapter_number": 2,
            "chapter_title": "The Guild",
            "events": [
                {
                    "description": "Mira joined the Guild to protect Arn.",
                    "event_type": "relationship",
                    "participants": ["Mira", "Guild"],
                    "evidence": "Mira swore herself to the Guild.",
                }
            ],
        },
    )
