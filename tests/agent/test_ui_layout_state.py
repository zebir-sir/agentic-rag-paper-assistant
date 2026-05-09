from ui.layout_state import should_show_welcome_guide_from_state


def test_should_show_welcome_guide_from_state_empty_state():
    assert should_show_welcome_guide_from_state([], None, False, "") is True


def test_should_show_welcome_guide_from_state_messages_hide_welcome():
    assert should_show_welcome_guide_from_state([{"role": "user", "content": "hi"}], None, False, "") is False


def test_should_show_welcome_guide_from_state_pending_prompt_hides_welcome():
    assert should_show_welcome_guide_from_state([], "总结这篇论文", False, "") is False


def test_should_show_welcome_guide_from_state_streaming_hides_welcome():
    assert should_show_welcome_guide_from_state([], None, True, "") is False


def test_should_show_welcome_guide_from_state_partial_streaming_response_hides_welcome():
    assert should_show_welcome_guide_from_state([], None, False, "partial") is False
