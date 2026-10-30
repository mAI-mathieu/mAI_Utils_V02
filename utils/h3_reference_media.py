"""Ordered H3 references and a confined, persistent media store (no ComfyUI imports)."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


MEDIA_DIRECTORY = Path(__file__).resolve().parents[1] / "reference_media"
MEDIA_EXTENSIONS = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    "audio": (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"),
}
REFERENCE_LIMITS = {"image": 9, "audio": 3, "video": 3, "total": 15}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class H3MediaReference:
    # Internal descriptor; the node decodes these into native IMAGE/AUDIO sockets.
    kind: str
    path: str


def media_kind(filename):
    suffix = Path(filename).suffix.lower()
    for kind, extensions in MEDIA_EXTENSIONS.items():
        if suffix in extensions:
            return kind
    raise ValueError(f"Unsupported reference media extension: {suffix or '(none)'}")


def media_path(filename, root=MEDIA_DIRECTORY, *, must_exist=True):
    """Only content-addressed files in the media store may be served or executed."""
    if not isinstance(filename, str) or not re.fullmatch(r"[0-9a-f]{64}\.[a-z0-9]+", filename):
        raise ValueError("Invalid reference filename; add the file through the reference node.")
    media_kind(filename)
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reference media must remain inside its media folder.")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Reference media is missing: {filename}. Add the original file again.")
    return path


def parse_manifest(value):
    try:
        entries = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("media_manifest must be a JSON array of uploaded media entries.") from exc
    if not isinstance(entries, list):
        raise ValueError("media_manifest must be a JSON array.")
    if len(entries) > REFERENCE_LIMITS["total"]:
        raise ValueError("H3 supports at most 9 images, 3 videos and 3 standalone audio references.")
    counts = dict.fromkeys(MEDIA_EXTENSIONS, 0)
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each reference must contain filename, name and kind.")
        filename = entry.get("filename")
        # Validate even before a file is looked up.
        media_path(filename, must_exist=False)
        kind = media_kind(filename)
        if entry.get("kind") != kind:
            raise ValueError(f"Reference kind does not match its file extension: {filename}")
        name = entry.get("name", filename)
        if not isinstance(name, str) or not name.strip() or len(name) > 512:
            raise ValueError("Reference display names must contain 1 to 512 characters.")
        counts[kind] += 1
        if counts[kind] > REFERENCE_LIMITS[kind]:
            raise ValueError(f"H3 supports at most {REFERENCE_LIMITS[kind]} {kind} references.")
        result.append({"filename": filename, "name": name, "kind": kind})
    return result


def resolve_references(value, root=MEDIA_DIRECTORY):
    entries = parse_manifest(value)
    references = tuple(H3MediaReference(item["kind"], str(media_path(item["filename"], root))) for item in entries)
    return references, entries


def reference_fingerprint(value, root=MEDIA_DIRECTORY):
    """Order and on-disk changes both invalidate ComfyUI's execution cache."""
    references, _ = resolve_references(value, root)
    return tuple((ref.kind, ref.path, Path(ref.path).stat().st_size,
                  Path(ref.path).stat().st_mtime_ns) for ref in references)


