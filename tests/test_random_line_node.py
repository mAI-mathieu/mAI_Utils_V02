from nodes.random_line_node import MAIRandomLine


def run_random_line(text):
    node = MAIRandomLine()
    return node.run(text)


def test_multiline_text_returns_valid_non_empty_line(monkeypatch):
    selected = []

    def choose(lines):
        selected.append(lines)
        return lines[1]

    monkeypatch.setattr("nodes.random_line_node.random.choice", choose)

    result = run_random_line("Line 1\nLine 2\nLine 3")

    assert result == ("Line 2",)
    assert selected == [["Line 1", "Line 2", "Line 3"]]


def test_empty_input_returns_empty_string():
    assert run_random_line("") == ("",)


def test_whitespace_only_input_returns_empty_string():
    assert run_random_line("   \n\t\r\n") == ("",)


def test_empty_lines_are_ignored(monkeypatch):
    selected = []

    def choose(lines):
        selected.append(lines)
        return lines[0]

    monkeypatch.setattr("nodes.random_line_node.random.choice", choose)

    result = run_random_line("\nFirst\n\nSecond\n")

    assert result == ("First",)
    assert selected == [["First", "Second"]]


def test_windows_crlf_line_endings_work(monkeypatch):
    selected = []

    def choose(lines):
        selected.append(lines)
        return lines[-1]

    monkeypatch.setattr("nodes.random_line_node.random.choice", choose)

    result = run_random_line("First\r\nSecond\r\n")

    assert result == ("Second",)
    assert selected == [["First", "Second"]]
