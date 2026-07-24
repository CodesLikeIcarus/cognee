"""Loader that drives the hosted Unstructured Transform MCP server.

Same managed partitioning pipeline as the REST-based Transform loader, but
spoken over the Model Context Protocol (streamable HTTP). Useful when a
deployment already fronts document processing with the Transform MCP server
and wants cognee ingestion to go through that single endpoint. The server
renders results to markdown for us, so no client-side element handling is
needed.
"""

import asyncio
import json
import mimetypes
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.loaders.external.unstructured_transform_loader import (
    API_KEY_SIGNUP_URL,
    MAX_FILE_SIZE_BYTES,
    PARTITION_STRATEGIES,
    get_unstructured_transform_settings,
)
from cognee.shared.logging_utils import get_logger

try:
    from mcp import ClientSession  # ty:ignore[unresolved-import]
    from mcp.client.streamable_http import streamablehttp_client  # ty:ignore[unresolved-import]
except ImportError as e:
    raise ImportError(
        "mcp is required for UnstructuredTransformMcpLoader. "
        "Install with: pip install 'cognee[unstructured-transform-mcp]'"
    ) from e

logger = get_logger(__name__)

TERMINAL_JOB_STATUSES = ("COMPLETED", "FAILED", "STOPPED")


@asynccontextmanager
async def _open_session(server_url: str, api_key: str):
    """Open an initialized MCP session against the Transform server."""
    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(server_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _tool_payload(result: Any) -> dict[str, Any]:
    """Extract the JSON payload from an MCP tool result.

    FastMCP publishes dict results as structured content; fall back to parsing
    the text block for servers/transports that only emit text content.
    """
    payload = getattr(result, "structuredContent", None)
    if payload is None:
        content = getattr(result, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                payload = json.loads(text)
                break
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected Transform MCP tool result: no JSON payload found.")
    # FastMCP wraps non-object returns under a "result" key; unwrap for uniformity.
    if set(payload.keys()) == {"result"} and isinstance(payload["result"], dict):
        payload = payload["result"]
    return payload


def _raise_on_error(payload: dict[str, Any], context: str) -> dict[str, Any]:
    error = payload.get("error")
    if error:
        code = error.get("code", "unknown_error")
        message = error.get("message", "")
        raise RuntimeError(f"Transform MCP {context} failed ({code}): {message}")
    return payload


async def _upload_file(upload_url: str, headers: dict[str, str], data: bytes) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.put(upload_url, headers=headers, content=data)
        response.raise_for_status()


async def _download_text(download_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(download_url)
        response.raise_for_status()
        return response.text


class UnstructuredTransformMcpLoader(LoaderInterface):
    """Partition documents via the Transform MCP server's job tools.

    Uploads the file through ``request_file_upload_url``, starts a
    partition-only job with ``start_transform_job``, polls
    ``check_job_status``, and fetches the markdown render from
    ``get_job_results``. Activates only when
    ``UNSTRUCTURED_TRANSFORM_API_KEY`` is set (keys: see
    https://transform.unstructured.io/get-started).
    """

    loader_name = "unstructured_transform_mcp_loader"

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
        output_format: str = "md",
        api_key: Optional[str] = None,
        mcp_url: Optional[str] = None,
        job_timeout: float = 900.0,
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
        mcp_url = mcp_url or settings.unstructured_transform_mcp_url
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
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        async with _open_session(mcp_url, api_key) as session:
            text = await self._partition_via_mcp(
                session,
                file_name=file_name,
                content_type=content_type,
                file_content=file_content,
                strategy=strategy,
                output_format=output_format,
                job_timeout=job_timeout,
            )

        if not kwargs.get("persist", True):
            return text

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        return await storage.store(storage_file_name, text)

    async def _partition_via_mcp(
        self,
        session: "ClientSession",
        *,
        file_name: str,
        content_type: str,
        file_content: bytes,
        strategy: str,
        output_format: str,
        job_timeout: float,
    ) -> str:
        upload = _raise_on_error(
            _tool_payload(
                await session.call_tool(
                    "request_file_upload_url",
                    {
                        "filename": file_name,
                        "content_type": content_type,
                        "size_bytes": len(file_content),
                    },
                )
            ),
            "upload-url request",
        )
        await _upload_file(upload["upload_url"], upload.get("headers", {}), file_content)

        job = _raise_on_error(
            _tool_payload(
                await session.call_tool(
                    "start_transform_job",
                    {
                        "file_refs": [upload["file_ref"]],
                        "stages": {"partition": {"strategy": strategy}},
                    },
                )
            ),
            "job start",
        )
        job_id = job["job_id"]
        logger.info(f"Submitted Transform MCP job {job_id} for {file_name}")

        status = job.get("status", "SCHEDULED")
        deadline = asyncio.get_event_loop().time() + job_timeout
        while status not in TERMINAL_JOB_STATUSES:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Transform MCP job {job_id} did not finish within {job_timeout} seconds."
                )
            snapshot = _raise_on_error(
                _tool_payload(await session.call_tool("check_job_status", {"job_id": job_id})),
                "status check",
            )
            status = snapshot.get("status", status)
            if status not in TERMINAL_JOB_STATUSES:
                await asyncio.sleep(float(snapshot.get("poll_after") or 5))

        if status != "COMPLETED":
            raise RuntimeError(f"Transform MCP job {job_id} ended with status {status}.")

        results = _raise_on_error(
            _tool_payload(
                await session.call_tool(
                    "get_job_results",
                    {"job_id": job_id, "output_format": output_format},
                )
            ),
            "results fetch",
        )
        files = results.get("files") or []
        if not files:
            raise RuntimeError(f"Transform MCP job {job_id} completed but returned no files.")
        entry = files[0]
        if entry.get("content"):
            return entry["content"]
        download_url = entry.get("download_url")
        if not download_url:
            raise RuntimeError(
                f"Transform MCP job {job_id} returned neither inline content nor a download URL."
            )
        return await _download_text(download_url)
