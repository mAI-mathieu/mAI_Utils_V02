import json
from pathlib import Path
import shutil
import subprocess

import pytest

from utils.h3_reference_media import (
    media_kind, media_path, parse_manifest, reference_fingerprint, resolve_references,
)
from nodes.h3_media_references import MAIH3MediaReferences


def entry(index=0, kind="image"):
    suffix = {"image": ".png", "audio": ".wav", "video": ".mp4"}[kind]
    return {"filename": f"{index:064x}{suffix}", "name": f"reference {index}{suffix}", "kind": kind}


def test_manifest_order_duplicates_and_empty(tmp_path):
    entries = [entry(2), entry(1, "audio"), entry(2), entry(3, "video")]
    for item in entries:
        (tmp_path / item["filename"]).write_bytes(b"media")
    references, parsed = resolve_references(json.dumps(entries), tmp_path)
    assert parsed == entries
    assert [ref.kind for ref in references] == ["image", "audio", "image", "video"]
    assert [Path(ref.path).name for ref in references] == [item["filename"] for item in entries]
    assert resolve_references("[]", tmp_path) == ((), [])
    assert MAIH3MediaReferences().run() == (*([None] * 18), "[]")


@pytest.mark.parametrize("value", ["null", "{}", "[1]", "invalid", None, '[{"filename":"../escape.png"}]',
                                       json.dumps([{**entry(), "kind": "audio"}]),
                                       json.dumps([{**entry(), "name": ""}])])
def test_invalid_manifests(value):
    with pytest.raises(ValueError):
        parse_manifest(value)


@pytest.mark.parametrize("kind,limit", [("image", 9), ("video", 3), ("audio", 3)])
def test_per_kind_limits(kind, limit):
    assert len(parse_manifest(json.dumps([entry(i, kind) for i in range(limit)]))) == limit
    with pytest.raises(ValueError, match="H3 supports"):
        parse_manifest(json.dumps([entry(i, kind) for i in range(limit + 1)]))


def test_all_fifteen_native_slots_are_supported():
    values = [entry(i, kind) for kind, count in (("image", 9), ("video", 3), ("audio", 3)) for i in range(count)]
    assert len(parse_manifest(json.dumps(values))) == 15


@pytest.mark.parametrize("filename", ["../a.png", "C:/a.png", "/a.png", "https://a/b.png", "a.png", "f" * 64 + ".html", "f" * 64 + "/a.png", None])
def test_path_rejects_unmanaged_files(tmp_path, filename):
    with pytest.raises(ValueError):
        media_path(filename, tmp_path)


def test_missing_media_explains_recovery(tmp_path):
    with pytest.raises(FileNotFoundError, match="Add the original file again"):
        resolve_references(json.dumps([entry()]), tmp_path)


def test_fingerprint_changes_when_file_or_order_changes(tmp_path):
    values = [entry(1), entry(2)]
    for item in values:
        (tmp_path / item["filename"]).write_bytes(b"one")
    original = reference_fingerprint(json.dumps(values), tmp_path)
    assert original != reference_fingerprint(json.dumps(values[::-1]), tmp_path)
    (tmp_path / values[0]["filename"]).write_bytes(b"replacement")
    assert original != reference_fingerprint(json.dumps(values), tmp_path)


def test_native_output_contract():
    node = MAIH3MediaReferences
    assert node.CATEGORY == "mAI / Input"
    assert node.RETURN_TYPES == ("IMAGE",) * 12 + ("AUDIO",) * 6 + ("STRING",)
    assert node.RETURN_NAMES == (tuple(f"ref_image_{i}" for i in range(9)) + tuple(f"ref_video_{i}" for i in range(3))
                                 + tuple(f"ref_video_audio_{i}" for i in range(3)) + tuple(f"ref_audio_{i}" for i in range(3)) + ("manifest",))
    assert media_kind("UPPER.JPG") == "image"


def test_pack_registration_preserves_existing_nodes(monkeypatch):
    import ast
    import importlib.util
    import sys

    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    name = "mai_h3_media_registration"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIH3MediaReferences"].__name__ == "MAIH3MediaReferences"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIH3MediaReferences"] == "mAI H3 media references"
    assert package.WEB_DIRECTORY == "./web"
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    mapping = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in node.targets))
    assert {key.value for key in mapping.keys} <= package.NODE_CLASS_MAPPINGS.keys()


def test_frontend_behaviors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run([node, str(Path(__file__).with_name("h3_media_references_frontend.mjs"))], capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
