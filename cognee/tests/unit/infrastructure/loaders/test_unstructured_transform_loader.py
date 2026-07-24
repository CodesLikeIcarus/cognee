"""Unit tests for the Unstructured Transform loader.

All network interaction is mocked at the SDK-client boundary; no API key or
network access is needed.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("unstructured_client")

from unstructured_client.models.shared import JobStatus

import cognee.infrastructure.loaders.external.unstructured_transform_loader as loader_module
from cognee.infrastructure.loaders.external.unstructured_transform_loader import (
    UnstructuredTransformLoader,
    UnstructuredTransformSettings,
    _build_request_data,
    _elements_to_text,
    _normalize_server_url,
)


def _settings(api_key=None, api_url="https://platform-api.transform.unstructured.io"):
    return UnstructuredTransformSettings(
        unstructured_transform_api_key=api_key,
        unstructured_transform_api_url=api_url,
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


class FakeJobsClient:
    """Replays a scripted job lifecycle and records SDK calls."""

    def __init__(self, statuses, elements=None, failed_files=None):
        self.statuses = list(statuses)
        self.elements = elements if elements is not None else []
        self.failed_files = failed_files or []
        self.create_requests = []
        self.download_requests = []

    def _job(self, status):
        return SimpleNamespace(
            id="job-123",
            status=status,
            output_node_files=[SimpleNamespace(file_id="file-abc", node_id="node-1")],
        )

    async def create_job_async(self, request):
        self.create_requests.append(request)
        return SimpleNamespace(job_information=self._job(self.statuses.pop(0)))

    async def get_job_async(self, request):
        return SimpleNamespace(job_information=self._job(self.statuses.pop(0)))

    async def download_job_output_async(self, request):
        self.download_requests.append(request)
        return SimpleNamespace(any=self.elements)

    async def get_job_failed_files_async(self, request):
        return SimpleNamespace(job_failed_files=SimpleNamespace(failed_files=self.failed_files))


def _install_fake_client(monkeypatch, jobs_client):
    monkeypatch.setattr(
        loader_module,
        "_make_client",
        lambda api_key, api_url, http_client: SimpleNamespace(jobs=jobs_client),
    )


def test_can_handle_requires_api_key(unconfigured_settings):
    loader = UnstructuredTransformLoader()
    assert loader.can_handle("pdf", "application/pdf") is False


def test_can_handle_with_api_key(configured_settings):
    loader = UnstructuredTransformLoader()
    assert loader.can_handle("pdf", "application/pdf") is True
    assert loader.can_handle("docx", "application/octet-stream") is True
    assert loader.can_handle("txt", "text/plain") is False
    assert loader.can_handle("", "application/pdf") is False


def test_loader_is_registered():
    from cognee.infrastructure.loaders.supported_loaders import supported_loaders

    assert supported_loaders["unstructured_transform_loader"] is UnstructuredTransformLoader


def test_loader_is_in_default_priority_before_local_loaders():
    from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine

    priority = LoaderEngine().default_loader_priority
    assert priority.index("unstructured_transform_loader") < priority.index("pypdf_loader")
    assert priority.index("text_loader") < priority.index("unstructured_transform_loader")


def test_normalize_server_url():
    host = "https://platform-api.transform.unstructured.io"
    assert _normalize_server_url(host) == host
    assert _normalize_server_url(host + "/") == host
    assert _normalize_server_url(host + "/api/v1") == host
    assert _normalize_server_url(host + "/api/v1/") == host


def test_build_request_data_defaults_suppress_image_payloads():
    request_data = json.loads(_build_request_data("auto"))
    (partition_node,) = request_data["job_nodes"]
    assert partition_node["type"] == "partition"
    assert partition_node["subtype"] == "unstructured_api"
    assert partition_node["settings"]["strategy"] == "auto"
    assert partition_node["settings"]["extract_image_block_types"] == []
    assert partition_node["settings"]["extract_image_block_to_payload"] is False


def test_build_request_data_vlm():
    request_data = json.loads(_build_request_data("vlm"))
    (partition_node,) = request_data["job_nodes"]
    assert partition_node["subtype"] == "vlm"
    assert partition_node["settings"] == {"is_dynamic": True}


def test_elements_to_text_prefers_table_html_and_skips_empty():
    elements = [
        {"type": "Title", "text": "Report"},
        {"type": "NarrativeText", "text": ""},
        {
            "type": "Table",
            "text": "a b",
            "metadata": {"text_as_html": "<table><tr><td>a</td><td>b</td></tr></table>"},
        },
    ]
    text = _elements_to_text(elements)
    assert text == "Report\n\n<table><tr><td>a</td><td>b</td></tr></table>"


@pytest.mark.asyncio
async def test_load_returns_flattened_text(tmp_path, monkeypatch, configured_settings):
    jobs_client = FakeJobsClient(
        statuses=[JobStatus.SCHEDULED, JobStatus.IN_PROGRESS, JobStatus.COMPLETED],
        elements=[
            {"type": "Title", "text": "Hello"},
            {"type": "NarrativeText", "text": "World"},
        ],
    )
    _install_fake_client(monkeypatch, jobs_client)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html><body>Hello World</body></html>")

    loader = UnstructuredTransformLoader()
    text = await loader.load(str(file_path), poll_interval=0, persist=False)

    assert text == "Hello\n\nWorld"

    (create_request,) = jobs_client.create_requests
    body = create_request["body_create_job"]
    assert body["input_files"][0]["file_name"] == "sample.html"
    request_data = json.loads(body["request_data"])
    assert request_data["job_nodes"][0]["type"] == "partition"

    (download_request,) = jobs_client.download_requests
    assert download_request == {"job_id": "job-123", "file_id": "file-abc"}


@pytest.mark.asyncio
async def test_load_parses_json_string_output(tmp_path, monkeypatch, configured_settings):
    jobs_client = FakeJobsClient(
        statuses=[JobStatus.COMPLETED],
        elements=json.dumps([{"type": "Title", "text": "Hello"}]),
    )
    _install_fake_client(monkeypatch, jobs_client)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformLoader()
    text = await loader.load(str(file_path), poll_interval=0, persist=False)

    assert text == "Hello"


@pytest.mark.asyncio
async def test_load_failed_job_raises_with_details(tmp_path, monkeypatch, configured_settings):
    jobs_client = FakeJobsClient(
        statuses=[JobStatus.SCHEDULED, JobStatus.FAILED],
        failed_files=[SimpleNamespace(document="sample.html", error="partition exploded")],
    )
    _install_fake_client(monkeypatch, jobs_client)

    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformLoader()
    with pytest.raises(RuntimeError, match="partition exploded"):
        await loader.load(str(file_path), poll_interval=0, persist=False)


@pytest.mark.asyncio
async def test_load_without_api_key_raises_actionable_error(tmp_path, unconfigured_settings):
    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformLoader()
    with pytest.raises(RuntimeError, match="UNSTRUCTURED_TRANSFORM_API_KEY"):
        await loader.load(str(file_path), persist=False)


@pytest.mark.asyncio
async def test_load_rejects_files_over_size_limit(tmp_path, monkeypatch, configured_settings):
    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")
    monkeypatch.setattr(
        loader_module.os.path,
        "getsize",
        lambda path: loader_module.MAX_FILE_SIZE_BYTES + 1,
    )

    loader = UnstructuredTransformLoader()
    with pytest.raises(ValueError, match="50 MB"):
        await loader.load(str(file_path), persist=False)


@pytest.mark.asyncio
async def test_load_rejects_unknown_strategy(tmp_path, configured_settings):
    file_path = tmp_path / "sample.html"
    file_path.write_text("<html></html>")

    loader = UnstructuredTransformLoader()
    with pytest.raises(ValueError, match="strategy"):
        await loader.load(str(file_path), strategy="turbo", persist=False)
