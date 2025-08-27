import asyncio, httpx
from typing import Any, Dict, List
from .load_api import Settings
from supabase import create_client, Client

settings = Settings()

try:
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    supabase = None

EMBED_MODEL = getattr(settings, 'EMBED_MODEL', 'text-embedding-3-small')

class SupabaseVectorStore:
    def __init__(self, supabase_client: Client, table_name: str = "documents"):
        self.client = supabase_client
        self.table_name = table_name

    async def add_documents(self, documents: List[Dict[str, Any]]):
        if not self.client:
            return
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            try:
                await asyncio.to_thread(self.client.table(self.table_name).upsert(batch).execute)
            except Exception as e:
                print("Insert error", e)

    async def create_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=40.0) as client:
            try:
                r = await client.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}, json={"model": EMBED_MODEL, "input": text})
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]
            except Exception as e:
                print("embed error", e)
                return []

    async def similarity_search(self, query_embedding: List[float], user_id: str, match_count: int = 5) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        try:
            result = await asyncio.to_thread(self.client.rpc("match_documents", {"query_embedding": query_embedding, "match_count": match_count, "p_user_id": user_id}).execute)
            return result.data or []
        except Exception as e:
            print("similarity search error", e)
            return []