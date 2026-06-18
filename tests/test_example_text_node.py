from nodes.example_text_node import MAIExampleTextNode


def test_example_text_node_strips_text():
    node = MAIExampleTextNode()
    result = node.run("  hello  ", True)
    assert result == ("hello",)


def test_example_text_node_can_keep_whitespace():
    node = MAIExampleTextNode()
    result = node.run("  hello  ", False)
    assert result == ("  hello  ",)
