from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    text = app_path.read_text(encoding="utf-8")

    toolbar_pos = text.find("render_input_toolbar(")
    chat_input_pos = text.find("st.chat_input(")
    assert toolbar_pos != -1, "render_input_toolbar not found"
    assert chat_input_pos != -1, "st.chat_input not found"
    assert toolbar_pos < chat_input_pos, "render_input_toolbar must appear before st.chat_input"

    title_count = text.count('st.title("科研论文阅读助手")')
    assert title_count == 1, "app title should only appear once"

    assert "if st.session_state.is_streaming:\n        st.stop()" not in text
    assert "if st.session_state.is_streaming:\n        return" not in text

    layout_state_path = Path(__file__).resolve().parents[1] / "ui" / "layout_state.py"
    assert layout_state_path.exists(), "ui/layout_state.py should exist"

    print("UI layout contract OK")


if __name__ == "__main__":
    main()
