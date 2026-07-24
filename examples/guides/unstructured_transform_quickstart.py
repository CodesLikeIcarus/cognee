"""Ingest documents through the hosted Unstructured Transform API.

The Unstructured Transform loader partitions documents (PDF, Office, HTML,
email, ...) with Unstructured's managed pipeline — layout understanding, table
structure, optional VLM-assisted partitioning — and feeds the resulting text
into cognee's normal remember/recall flow. Better document parsing means
higher-fidelity entities and relationships in the knowledge graph.

Prerequisites:
1. pip install 'cognee[unstructured-transform]'
2. A Transform API key (free tier: 15K pages/month):
   https://transform.unstructured.io/get-started
3. In your `.env` (alongside your LLM_API_KEY):
   UNSTRUCTURED_TRANSFORM_API_KEY="your-transform-api-key"

With the key set, the loader activates automatically for supported formats;
without it, cognee's local loaders handle files exactly as before.
"""

import asyncio
from os import path

import cognee
from cognee import SearchType

DOCUMENT_PATH = path.join(
    path.dirname(__file__),
    "..",
    "..",
    "cognee",
    "tests",
    "test_data",
    "artificial-intelligence.pdf",
)


async def main():
    await cognee.forget(everything=True)

    # The API key in the environment is enough: the Transform loader picks up
    # supported document formats automatically.
    await cognee.remember(DOCUMENT_PATH, self_improvement=False)

    # Alternatively, opt in explicitly and pick a partition strategy —
    # for example VLM-assisted partitioning for visually complex documents:
    # await cognee.remember(
    #     DOCUMENT_PATH,
    #     preferred_loaders={"unstructured_transform_loader": {"strategy": "vlm"}},
    #     self_improvement=False,
    # )

    results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is artificial intelligence?",
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
