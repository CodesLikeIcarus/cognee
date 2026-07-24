"""Integration test for the Unstructured Transform MCP loader.

Talks to the live hosted MCP server, so it only runs when
UNSTRUCTURED_TRANSFORM_API_KEY is set (free keys:
https://transform.unstructured.io/get-started).
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("unstructured_client")

from cognee.infrastructure.loaders.external.unstructured_transform_mcp_loader import (
    UnstructuredTransformMcpLoader,
)

requires_api_key = pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_TRANSFORM_API_KEY"),
    reason="UNSTRUCTURED_TRANSFORM_API_KEY not set",
)

TEST_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


@requires_api_key
@pytest.mark.asyncio
async def test_partitions_docx_through_mcp_server():
    loader = UnstructuredTransformMcpLoader()

    text = await loader.load(str(TEST_DATA_DIR / "example.docx"), persist=False)

    assert isinstance(text, str)
    assert text.strip()
