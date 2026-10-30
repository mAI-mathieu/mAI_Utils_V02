import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from utils import h3_reference_routes as routes


def run(coro):
    return asyncio.run(coro)


class Part:
    name = "file"

    def __init__(self, filename, chunks):
        self.filename = filename
        self.chunks = iter(chunks)

    async def read_chunk(self, size):
        return next(self.chunks, b"")


class Request:
    def __init__(self, part):
        self.part = part

    async def multipart(self):
        return self

    async def next(self):
        return self.part


def test_upload_deduplicates_and_preserves_name(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)
    payload = b"test reference media"
    expected = hashlib.sha256(payload).hexdigest() + ".png"
    for _ in range(2):
        response = run(routes.upload_reference(Request(Part("../Photo.PNG", [payload[:4], payload[4:]]))))
        assert response.status == 200
        assert json.loads(response.body) == {"filename": expected, "name": "Photo.PNG", "kind": "image"}
    assert [path.name for path in tmp_path.iterdir()] == [expected]
    assert (tmp_path / expected).read_bytes() == payload


def test_identical_media_preserves_each_exact_original_name(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)
    names = ["  Shot 0002 - ÉTÉ (Final).PNG", "Shot 0010 Original.PNG"]
    responses = [json.loads(run(routes.upload_reference(Request(Part(name, [b"same media"])))).body) for name in names]
    assert [item["name"] for item in responses] == names
    assert responses[0]["filename"] == responses[1]["filename"]


@pytest.mark.parametrize("name,chunks", [("evil.html", [b"bad"]), ("empty.mp4", []), ("x" * 513 + ".png", [b"bad"])])
def test_invalid_uploads_leave_no_partial_file(tmp_path, monkeypatch, name, chunks):
    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)
    response = run(routes.upload_reference(Request(Part(name, chunks))))
    assert response.status == 400
    assert not list(tmp_path.iterdir())


def test_oversize_upload_cleans_partial_file(tmp_path, monkeypatch):
    from aiohttp import web

    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 2)
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        run(routes.upload_reference(Request(Part("big.mp4", [b"123"]))))
    assert not list(tmp_path.iterdir())


def test_preview_rejects_traversal_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)
    assert run(routes.preview_reference(SimpleNamespace(match_info={"filename": "../secret.png"}))).status == 400
    assert run(routes.preview_reference(SimpleNamespace(match_info={"filename": "a" * 64 + ".png"}))).status == 404


def test_routes_register_once_and_skip_offline(monkeypatch):
    import sys
    from aiohttp import web

    monkeypatch.delitem(sys.modules, "server", raising=False)
    routes.register_reference_routes()
    instance = SimpleNamespace(routes=web.RouteTableDef())
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=instance)))
    routes.register_reference_routes()
    routes.register_reference_routes()
    assert len(instance.routes) == 3


def test_http_upload_preview_and_range_seeking(tmp_path, monkeypatch):
    from aiohttp import FormData, web
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setattr(routes, "MEDIA_DIRECTORY", tmp_path)

    async def exercise():
        app = web.Application()
        app.router.add_post("/upload", routes.upload_reference)
        app.router.add_get("/media/{filename}", routes.preview_reference)
        app.router.add_get("/settings", routes.reference_settings)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/settings")
            assert (await response.json())["limits"]["image"] == 9
            data = FormData()
            data.add_field("file", b"0123456789", filename="movie.mp4", content_type="video/mp4")
            response = await client.post("/upload", data=data)
            assert response.status == 200
            filename = (await response.json())["filename"]
            response = await client.get(f"/media/{filename}", headers={"Range": "bytes=3-6"})
            assert response.status == 206
            assert await response.read() == b"3456"
            assert response.headers["Content-Type"] == "video/mp4"
            assert response.headers["X-Content-Type-Options"] == "nosniff"

    run(exercise())
