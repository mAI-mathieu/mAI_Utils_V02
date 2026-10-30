"""ComfyUI-local upload and range-capable preview routes for reference media."""

import asyncio
import hashlib
from pathlib import Path
import sys
import tempfile

from .h3_reference_media import (
    MAX_UPLOAD_BYTES, MEDIA_DIRECTORY, MEDIA_EXTENSIONS, REFERENCE_LIMITS,
    media_kind, media_path,
)


async def upload_reference(request):
    from aiohttp import web

    temporary = None
    try:
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file" or not part.filename:
            raise ValueError("Upload one media file using the 'file' field.")
        name = part.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not name.strip() or len(name) > 512:
            raise ValueError("Media filenames must contain 1 to 512 characters.")
        kind = media_kind(name)
        suffix = Path(name).suffix.lower()
        MEDIA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(dir=MEDIA_DIRECTORY, prefix=".upload_", delete=False) as stream:
            temporary = Path(stream.name)
            while chunk := await part.read_chunk(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD_BYTES, actual_size=size)
                digest.update(chunk)
                await asyncio.to_thread(stream.write, chunk)
        if size == 0:
            raise ValueError("The reference file is empty.")
        filename = digest.hexdigest() + suffix
        destination = media_path(filename, MEDIA_DIRECTORY, must_exist=False)
        # Identical uploads share storage; names and ordering remain per-node.
        if destination.exists():
            temporary.unlink()
        else:
            await asyncio.to_thread(temporary.replace, destination)
        temporary = None
        return web.json_response({"filename": filename, "name": name, "kind": kind})
    except (ValueError, FileNotFoundError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def preview_reference(request):
    from aiohttp import web

    try:
        path = media_path(request.match_info["filename"], MEDIA_DIRECTORY)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    # FileResponse supports byte ranges for seeking in large audio/video files.
    return web.FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})


async def reference_settings(request):
    from aiohttp import web

    return web.json_response({"extensions": MEDIA_EXTENSIONS, "limits": REFERENCE_LIMITS,
                              "max_upload_bytes": MAX_UPLOAD_BYTES})


def register_reference_routes():
    """Skip registration outside ComfyUI (pure tests and offline tooling)."""
    server = sys.modules.get("server")
    instance = getattr(getattr(server, "PromptServer", None), "instance", None)
    if instance is None or getattr(instance, "_mai_h3_reference_routes", False):
        return
    instance.routes.post("/mai/h3/references/upload")(upload_reference)
    instance.routes.get("/mai/h3/references/media/{filename}")(preview_reference)
    instance.routes.get("/mai/h3/references/settings")(reference_settings)
    instance._mai_h3_reference_routes = True
