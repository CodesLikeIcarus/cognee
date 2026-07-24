"""Integration test for the Unstructured Transform loader.

Talks to the live hosted API, so it only runs when UNSTRUCTURED_TRANSFORM_API_KEY
is set (free keys: https://transform.unstructured.io/get-started).
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("unstructured_client")

from cognee.infrastructure.loaders.external.unstructured_transform_loader import (
    UnstructuredTransformLoader,
)

requires_api_key = pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_TRANSFORM_API_KEY"),
    reason="UNSTRUCTURED_TRANSFORM_API_KEY not set",
)

TEST_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


@requires_api_key
@pytest.mark.asyncio
async def test_partitions_docx_through_hosted_api():
    loader = UnstructuredTransformLoader()

    text = await loader.load(str(TEST_DATA_DIR / "example.docx"), persist=False)

    assert isinstance(text, str)
    assert text.strip()


@requires_api_key
@pytest.mark.asyncio
async def test_partitions_pdf_through_hosted_api():
    loader = UnstructuredTransformLoader()

    text = await loader.load(str(TEST_DATA_DIR / "artificial-intelligence.pdf"), persist=False)

    assert isinstance(text, str)
    assert "artificial intelligence" in text.lower()
