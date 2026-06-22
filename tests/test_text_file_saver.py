from pathlib import Path

import pytest

from utils.text_file_saver import save_text_file


def test_saves_utf8_text_exactly(tmp_path):
    text = "Cafe\u00e9, \u010cesk\u00fd text, \u65e5\u672c\u8a9e, \u041f\u0440\u0438\u0432\u0435\u0442"

    saved_path = save_text_file(text, "output.txt", str(tmp_path))

    output_path = Path(saved_path)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == text


def test_overwrite_false_raises_when_file_exists(tmp_path):
    output_path = tmp_path / "output.txt"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_text_file("new text", "output.txt", str(tmp_path), overwrite=False)


@pytest.mark.parametrize("file_name", ["", "   "])
def test_empty_file_name_raises_value_error(tmp_path, file_name):
    with pytest.raises(ValueError):
        save_text_file("text", file_name, str(tmp_path))


@pytest.mark.parametrize(
    "file_name",
    [
        "../test.txt",
        "folder/test.txt",
        "folder\\test.txt",
        "/tmp/test.txt",
        "C:\\tmp\\test.txt",
        "..",
    ],
)
def test_path_traversal_in_file_name_raises_value_error(tmp_path, file_name):
    with pytest.raises(ValueError):
        save_text_file("text", file_name, str(tmp_path))
