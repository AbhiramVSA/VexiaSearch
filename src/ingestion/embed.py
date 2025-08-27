from typing import List
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ..load_api import Settings

settings = Settings()

MODEL = settings.__dict__.get("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 128

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _embed_batch(client: httpx.AsyncClient, texts: List[str]) -> List[List[float]]:
    resp = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        json={"model": MODEL, "input": texts},
        timeout=40.0,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [d["embedding"] for d in data]

async def embed_texts(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            embs = await _embed_batch(client, batch)
            out.extend(embs)
    return out

__all__ = ["embed_texts"]
