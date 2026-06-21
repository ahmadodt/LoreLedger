from pathlib import Path


def test_streamlit_wires_relationship_graph_controls():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "Relationship Viewer" in source
    assert "Include Relationship Graph" in source
    assert "include_graph=include_graph" in source
    assert "render_relationship_edges(query_graph(base_dir, selected_graph_name))" in source
