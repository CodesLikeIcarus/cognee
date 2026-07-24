"""Unit tests for the Unstructured Transform MCP loader.

All MCP and HTTP interaction is mocked; no API key or network access is needed.
"""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
pytest.importorskip("unstructured_client")

import cognee.infrastructure.loaders.external.unstructured_transform_mcp_loader as loader_module
from cognee.infrastructure.loaders.external.unstructured_transform_loader import (
    UnstructuredTransformSettings,
)
from cognee.infrastructure.loaders.external.unstructured_transform_mcp_loader import (
    UnstructuredTransformMcpLoader,
    _tool_payload,
)


def _settings(api_key=None):
    return UnstructuredTransformSettings(
        unstructured_transform_api_key=api_key,
        _env_file=None,
    )


@pytest.fixture
def configured_settings(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "get_unstructured_transform_settings",
        lambda: _settings(api_key="test-key"),
    )


@pytest.fixture
def unconfigured_settings(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "get_unstructured_transform_settings",
        lambda: _settings(api_key=None),
    )


class FakeSession:
    """Replays scripted tool payloads and records call_tool invocations."""

    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        payload = self.payloads[name]
        if isinstance(payload, list):
            payload = payload.pop(0)
        return SimpleNamespace(structuredContent=payload, content=[])


def _install_fakes(monkeypatch, session, uploads=None, downloads=None):
    uploads = uploads if uploads is not None else []
    downloads = downloads if downloads is not None else {}

    @asynccontextmanager
    async def fake_open_session(server_url, api_key):
        yield session

    async def fake_upload(upload_url, headers, data):
        uploads.append((upload_url, headers, data))

    async def fake_download(download_url):
        return downloads[download_url]

    monkeypatch.setattr(loader_module, "_open_session", fake_open_session)
    monkeypatch.setattr(loader_module, "_upload_file", fake_upload)
    monkeypatch.setattr(loader_module, "_download_text", fake_download)
    return uploads


UPLOAD_PAYLOAD = {
    "upload_url": "https://mcp.example/upload/f1?token=t",
    "method": "PUT",
    "headers": {"Content-Type": "text/html"},
    "file_ref": "u10d://file/f1",
}


def test_can_handle_requires_api_key(unconfigured_settings):
    loader = UnstructuredTransformMcpLoader()
    assert loader.can_handle("pdf", "application/pdf") is False


def test_can_handle_with_api_key(configured_settings):
    loader = UnstructuredTransformMcpLoader()
    assert loader.can_handle("pdf", "application/pdf") is True
    assert loader.can_handle("txt", "text/plain") is False


def test_loader_is_registered():
    from cognee.infrastructure.loaders.supported_loaders import supported_loaders

    assert supported_loaders["unstructured_transform_mcp_loader"] is UnstructuredTransformMcpLoader


def test_tool_payload_unwraps_structured_and_text_content():
    structured = SimpleNamespace(structuredContent={"job_id": "j1"}, content=[])
    assert _tool_payload(structured) == {"job_id": "j1"}

    text_only = SimpleNamespace(
        structuredContent=None,
        content=[SimpleNamespace(text=json.dumps({"job_id": "j2"}))],
    )
    assert _tool_payload(text_only) == {"job_id": "j2"}

    wrapped = SimpleNamespace(structuredContent={"result": {"job_id": "j3"}}, content=[])
    assert _tool_payload(wrapped) == {"job_id": "j3"}


@pytest.mark.asyncio
async def test_load_uploads_starts_polls_and_downloads(tmp_path, monkeypatch, configured_settings):
    session = FakeSession(
        {
            "request_file_upload_url": UPLOAD_PAYLOAD,
            "start_transform_job": {"job_id": "job-9", "status": "SCHEDULED"},
            "check_job_status": [
                {"job_id": "job-9", "status": "IN_PROGRESS", "poll_after": 0},
                {"job_id": "job-9", "status": "COMPLETED"},
            ],
            "get_job_results": {
                "job_id": "job-9",
                "files": [{"file_id": "f1", "download_url": "https://mcp.example/output/a1"}],
            },
        }
    )
    uploads = _install_fakes(
        monkeypatch,
        session,
        downloads={"https://mcp.example/output/a1": "# Rendered\n\nmarkdown"},
    )

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformMcpLoader()
    text = await loader.load(str(file_path), persist=False)

    assert text == "# Rendered\n\nmarkdown"

    (upload_call,) = uploads
    assert upload_call[0] == UPLOAD_PAYLOAD["upload_url"]
    assert upload_call[1] == {"Content-Type": "text/html"}

    tool_names = [name for name, _ in session.calls]
    assert tool_names == [
        "request_file_upload_url",
        "start_transform_job",
        "check_job_status",
        "check_job_status",
        "get_job_results",
    ]
    start_args = dict(session.calls)["start_transform_job"]
    assert start_args["file_refs"] == ["u10d://file/f1"]
    assert start_args["stages"] == {"partition": {"strategy": "auto"}}


@pytest.mark.asyncio
async def test_load_uses_inline_content_when_present(tmp_path, monkeypatch, configured_settings):
    session = FakeSession(
        {
            "request_file_upload_url": UPLOAD_PAYLOAD,
            "start_transform_job": {"job_id": "job-9", "status": "COMPLETED"},
            "get_job_results": {
                "job_id": "job-9",
                "files": [{"file_id": "f1", "content": "inline markdown"}],
            },
        }
    )
    _install_fakes(monkeypatch, session)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformMcpLoader()
    text = await loader.load(str(file_path), persist=False)

    assert text == "inline markdown"


@pytest.mark.asyncio
async def test_load_raises_on_error_envelope(tmp_path, monkeypatch, configured_settings):
    session = FakeSession(
        {
            "request_file_upload_url": UPLOAD_PAYLOAD,
            "start_transform_job": {
                "error": {"code": "quota_exceeded", "message": "monthly page quota reached"}
            },
        }
    )
    _install_fakes(monkeypatch, session)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformMcpLoader()
    with pytest.raises(RuntimeError, match="quota_exceeded"):
        await loader.load(str(file_path), persist=False)


@pytest.mark.asyncio
async def test_load_raises_on_failed_job(tmp_path, monkeypatch, configured_settings):
    session = FakeSession(
        {
            "request_file_upload_url": UPLOAD_PAYLOAD,
            "start_transform_job": {"job_id": "job-9", "status": "SCHEDULED"},
            "check_job_status": {"job_id": "job-9", "status": "FAILED"},
        }
    )
    _install_fakes(monkeypatch, session)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformMcpLoader()
    with pytest.raises(RuntimeError, match="FAILED"):
        await loader.load(str(file_path), persist=False)


@pytest.mark.asyncio
async def test_load_without_api_key_raises_actionable_error(tmp_path, unconfigured_settings):
    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformMcpLoader()
    with pytest.raises(RuntimeError, match="UNSTRUCTURED_TRANSFORM_API_KEY"):
        await loader.load(str(file_path), persist=False)
