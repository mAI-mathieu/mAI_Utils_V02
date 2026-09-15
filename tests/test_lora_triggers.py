import pytest

from utils.lora_triggers import prompt_matches_triggers


@pytest.mark.parametrize(("prompt", "triggers", "expected"), [
    ("A CAT in a garden", "cat", True),
    ("a cathedral", "cat", False),
    ("cats scatter", "cat", False),
    ("(cat:1.2), portrait", "cat", True),
    ("a dog", "cat, dog", True),
    ("a dog", "cat\r\ndog", True),
    ("a bird", "cat, dog", False),
    ("fine\n  art portrait", " fine art ", True),
    ("fine modern art", "fine art", False),
    ("a sks_style portrait", "sks_style", True),
    ("a sks_style_extra portrait", "sks_style", False),
    ("a [style+] portrait", "[style+]", True),
    ("a style portrait", "[style+]", False),
    ("CAFÉ portrait", "café", True),
    ("a cat", " , \ncat,\n", True),
    ("a cat", " , \n\t", False),
    ("a cat", "", False),
    ("", "cat", False),
])
def test_prompt_matching(prompt, triggers, expected):
    assert prompt_matches_triggers(prompt, triggers) is expected
