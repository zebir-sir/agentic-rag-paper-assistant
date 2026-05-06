from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    text = (root / "ui" / "api_client.py").read_text(encoding="utf-8")
    assert "def ensure_chat_state(" in text, "ensure_chat_state not found in ui/api_client.py"
    print("PASS: ensure_chat_state exists")


if __name__ == "__main__":
    main()
