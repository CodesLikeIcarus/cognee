"""Loader backed by the hosted Unstructured Transform API.

Sends documents to Unstructured's managed partitioning pipeline (layout
understanding, table structure, optional VLM-assisted partitioning) and turns
the returned structured elements into text for cognee's ingestion flow. The
service is remote, so this loader activates only when an API key is
configured; without one it never matches and local loaders handle the file.
"""

import asyncio
import json
import os
from functools import lru_cache
from typing import Any, Optional

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

try:
    from unstructured_client import UnstructuredClient  # ty:ignore[unresolved-import]
    from unstructured_client.models.shared import JobStatus  # ty:ignore[unresolved-import]
except ImportError as e:
    raise ImportError(
        "unstructured-client is required for UnstructuredTransformLoader. "
        "Install with: pip install 'cognee[unstructured-transform]'"
    ) from e

logger = get_logger(__name__)

API_KEY_SIGNUP_URL = "https://transform.unstructured.io/get-started"

# Server-side ingestion cap; larger files are rejected by the API anyway, so
# fail fast with a clear message instead of uploading doomed bytes.
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

PARTITION_STRATEGIES = ("auto", "fast", "hi_res", "vlm")

TERMINAL_JOB_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED)


class UnstructuredTransformSettings(BaseSettings):
    unstructured_transform_api_key: Optional[str] = None
    unstructured_transform_api_url: str = "https://platform-api.transform.unstructured.io"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_unstructured_transform_settings() -> UnstructuredTransformSettings:
    return UnstructuredTransformSettings()


def _normalize_server_url(url: str) -> str:
    """Reduce a configured endpoint to the bare host the SDK expects.

    The SDK's operation paths already include ``/api/v1/...``, so a base URL
    that carries the prefix (a natural way to copy it from API docs) would
    double it. Accept either form.
    """
    url = url.rstrip("/")
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return url


def _make_client(api_key: str, api_url: str, http_client: httpx.AsyncClient) -> UnstructuredClient:
    """Build an SDK client on a caller-owned httpx client.

    Supplying the httpx client keeps its lifecycle here (closed by the
    caller's context manager) instead of in the SDK's GC finalizer, which
    would otherwise try to close it again at interpreter shutdown and emit
    an unawaited-coroutine warning.
    """
    return UnstructuredClient(
        api_key_auth=api_key,
        server_url=_normalize_server_url(api_url),
        async_client=http_client,
    )


def _build_request_data(strategy: str) -> str:
    """Build the job DAG: a single partition node.

    Cognee chunks and embeds downstream of the loader, so the job requests
    partitioning only. Base64 image extraction is explicitly disabled — the
    server default attaches image payloads to most element types, which
    inflates the response severalfold with bytes a text pipeline never reads.
    """
    if strategy == "vlm":
        # The VLM subtype resolves provider/model server-side.
        settings: dict[str, Any] = {"is_dynamic": True}
        subtype = "vlm"
    else:
        settings = {
            "strategy": strategy,
            "extract_image_block_types": [],
            "extract_image_block_to_payload": False,
        }
        subtype = "unstructured_api"

    partition_node = {
        "name": "Partitioner",
        "type": "partition",
        "subtype": subtype,
        "settings": settings,
    }
    return json.dumps({"job_nodes": [partition_node]})


def _elements_to_text(elements: list[dict[str, Any]]) -> str:
    """Flatten partitioned elements into text, keeping table structure as HTML."""
    parts: list[str] = []
    for element in elements:
        text = element.get("text") or ""
        if element.get("type") == "Table":
            html = (element.get("metadata") or {}).get("text_as_html")
            if html:
                text = html
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


class UnstructuredTransformLoader(LoaderInterface):
    """Partition documents through the hosted Unstructured Transform API.

    Submits the file as an on-demand job, polls until it completes, downloads
    the element JSON, and stores the flattened text. Activates only when
    ``UNSTRUCTURED_TRANSFORM_API_KEY`` is set (keys: see
    https://transform.unstructured.io/get-started); otherwise ``can_handle``
    returns False and file handling falls through to the local loaders.
    """

    loader_name = "unstructured_transform_loader"

    _SUPPORTED_EXTENSIONS = [
        "pdf",
        "doc",
        "docx",
        "odt",
        "ppt",
        "pptx",
        "odp",
        "xls",
        "xlsx",
        "ods",
        "rtf",
        "epub",
        "html",
        "htm",
        "eml",
        "msg",
    ]

    @property
    def supported_extensions(self) -> list[str]:
        return self._SUPPORTED_EXTENSIONS

    @property
    def supported_mime_types(self) -> list[str]:
        return ["*/*"]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        if not extension:
            return False
        if not get_unstructured_transform_settings().unstructured_transform_api_key:
            return False
        return extension.lower().lstrip(".") in self._SUPPORTED_EXTENSIONS

    async def load(
        self,
        file_path: str,
        strategy: str = "auto",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        job_timeout: float = 900.0,
        poll_interval: float = 5.0,
        **kwargs: Any,
    ) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if strategy not in PARTITION_STRATEGIES:
            raise ValueError(
                f"Unknown partition strategy {strategy!r}; expected one of {PARTITION_STRATEGIES}."
            )

        settings = get_unstructured_transform_settings()
        api_key = api_key or settings.unstructured_transform_api_key
        api_url = api_url or settings.unstructured_transform_api_url
        if not api_key:
            raise RuntimeError(
                "No Unstructured Transform API key configured. Set the "
                "UNSTRUCTURED_TRANSFORM_API_KEY environment variable or pass api_key. "
                f"Free API keys: {API_KEY_SIGNUP_URL}"
            )

        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File {file_path} is {file_size} bytes; the Unstructured Transform API "
                f"accepts files up to {MAX_FILE_SIZE_BYTES} bytes (50 MB)."
            )

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
            f.seek(0)
            file_content = f.read()

        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"
        file_name = os.path.basename(file_path)

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as http_client:
            client = _make_client(api_key, api_url, http_client)
            elements = await self._partition_remotely(
                client,
                file_name=file_name,
                file_content=file_content,
                strategy=strategy,
                job_timeout=job_timeout,
                poll_interval=poll_interval,
            )
        text = _elements_to_text(elements)

        if not kwargs.get("persist", True):
            return text

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        return await storage.store(storage_file_name, text)

    async def _partition_remotely(
        self,
        client: UnstructuredClient,
        *,
        file_name: str,
        file_content: bytes,
        strategy: str,
        job_timeout: float,
        poll_interval: float,
    ) -> list[dict[str, Any]]:
        create_response = await client.jobs.create_job_async(
            request={
                "body_create_job": {
                    "request_data": _build_request_data(strategy),
                    "input_files": [{"content": file_content, "file_name": file_name}],
                }
            }
        )
        job = create_response.job_information
        if job is None:
            raise RuntimeError("Unstructured Transform API returned no job information.")

        logger.info(f"Submitted Unstructured Transform job {job.id} for {file_name}")

        deadline = asyncio.get_event_loop().time() + job_timeout
        while job.status not in TERMINAL_JOB_STATUSES:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Unstructured Transform job {job.id} did not finish within "
                    f"{job_timeout} seconds."
                )
            await asyncio.sleep(poll_interval)
            poll_response = await client.jobs.get_job_async(request={"job_id": job.id})
            if poll_response.job_information is not None:
                job = poll_response.job_information

        if job.status is not JobStatus.COMPLETED:
            raise RuntimeError(await self._job_failure_message(client, job.id, job.status))

        output_files = job.output_node_files or []
        if not output_files:
            raise RuntimeError(
                f"Unstructured Transform job {job.id} completed but produced no output files."
            )

        download_response = await client.jobs.download_job_output_async(
            request={"job_id": job.id, "file_id": output_files[0].file_id}
        )
        elements = download_response.any
        if isinstance(elements, (str, bytes)):
            elements = json.loads(elements)
        if not isinstance(elements, list):
            raise RuntimeError(
                f"Unexpected output for Unstructured Transform job {job.id}: "
                "expected a JSON list of elements."
            )
        return elements

    async def _job_failure_message(
        self, client: UnstructuredClient, job_id: str, status: JobStatus
    ) -> str:
        message = f"Unstructured Transform job {job_id} ended with status {status.value}."
        try:
            failures_response = await client.jobs.get_job_failed_files_async(
                request={"job_id": job_id}
            )
            failed_files = (
                failures_response.job_failed_files.failed_files
                if failures_response.job_failed_files
                else []
            )
            details = "; ".join(f"{failure.document}: {failure.error}" for failure in failed_files)
            if details:
                message += f" Failed files: {details}"
        except Exception as error:
            logger.debug(f"Could not fetch failure details for job {job_id}: {error}")
        return message
