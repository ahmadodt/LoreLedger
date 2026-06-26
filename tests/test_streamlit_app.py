import ast
from pathlib import Path


def test_gpu_layers_input_accepts_all_layers_value():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "local_model_config"
    )
    gpu_layers_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "GPU layers"
    )
    keywords = {keyword.arg: keyword.value for keyword in gpu_layers_call.keywords}

    assert ast.literal_eval(keywords["min_value"]) == -1
    assert ast.literal_eval(keywords["value"]) == 20


def test_streamlit_wires_conversation_memory_controls():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'ASK_CONVERSATION_HISTORY_KEY = "ask_conversation_history"' in source
    assert "MAX_CONVERSATION_TURNS = 5" in source
    assert 'st.checkbox("Enable Conversation History", value=True)' in source
    assert "New Conversation" in source
    assert "reset_conversation_history()" in source
    assert "if conversation_enabled:" in source
    assert "render_conversation_history(conversation_history)" in source
    assert "conversation_history=conversation_history if conversation_enabled else None" in source
    assert "append_conversation_turn(question.strip(), result.answer)" in source


def test_streamlit_caps_conversation_history_to_five_turns():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "raw_history[-MAX_CONVERSATION_TURNS:]" in source
