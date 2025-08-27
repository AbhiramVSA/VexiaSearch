import asyncio
from src.rag import rerank

def test_rerank_no_docs():
    res = asyncio.run(rerank('query', []))
    assert res == []
