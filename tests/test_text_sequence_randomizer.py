import pytest

from nodes.text_sequence_randomizer import MAITextSequenceRandomizer
from utils.text_sequence import randomize_text_sequence


def test_fixed_seed_reproduces_the_same_comma_sequence():
    text = "text 1, text 2, text 3, text 4"

    first = randomize_text_sequence(text, "comma", 42)
    second = randomize_text_sequence(text, "comma", 42)

    assert first == second
    assert sorted(first.split(", ")) == ["text 1", "text 2", "text 3", "text 4"]


def test_different_seeds_can_produce_different_sequences():
    text = "one, two, three, four, five"

    assert randomize_text_sequence(text, "comma", 1) != randomize_text_sequence(
        text, "comma", 2
    )


def test_comma_mode_trims_items_and_ignores_empty_items():
    result = randomize_text_sequence(" first, ,second, third ", "comma", 0)

    assert sorted(result.split(", ")) == ["first", "second", "third"]


def test_line_break_mode_randomizes_non_empty_lines():
    result = randomize_text_sequence("first\n\n second \r\nthird", "line break", 7)

    assert sorted(result.splitlines()) == ["first", "second", "third"]


def test_empty_input_returns_empty_text():
    assert randomize_text_sequence("", "comma", 10) == ""


def test_unknown_separator_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown separator"):
        randomize_text_sequence("one, two", "semicolon", 0)


def test_seed_must_be_an_integer():
    with pytest.raises(TypeError, match="seed must be an integer"):
        randomize_text_sequence("one, two", "comma", "0")


def test_node_returns_a_single_text_output():
    node = MAITextSequenceRandomizer()

    assert node.run("one, two, three", "comma", 3) == (
        randomize_text_sequence("one, two, three", "comma", 3),
    )
