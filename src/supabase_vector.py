import asyncio
from load_api import Settings
from typing import Any, Dict, List

# Supabase and LangChain imports
from supabase_vector import create_client, Client
from langchain_openai import OpenAIEmbeddings

settings = Settings()

try:
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing Supabase or OpenAI Embeddings: {e}")
    print("Please ensure your API keys and URLs are set correctly.")
    supabase = None
    embeddings = None

class SupabaseVectorStore:
  
    def __init__(self, supabase_client: Client, table_name: str = "documents"):
        self.client = supabase_client
        self.table_name = table_name
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    async def add_documents(self, documents: List[Dict[str, Any]]):
       
        if not self.client:
            print("Supabase client not initialized. Skipping document addition.")
            return

        try:
            batch_size = 20
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = await asyncio.to_thread(
                    self.client.table(self.table_name).insert(batch).execute
                )
                if result.data:
                    print(f"Successfully inserted batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                else:
                    print(f"Error inserting batch: {getattr(result, 'error', 'Unknown error')}")
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error adding documents to Supabase: {e}")

    async def create_embedding(self, text: str) -> List[float]:
        
        if not self.embeddings:
            print("Embeddings client not initialized. Skipping embedding creation.")
            return []
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            # Added more detailed logging to show the failing text and error
            print(f"Error creating embedding for text chunk: '{text[:80]}...'. Error: {e}")
            return []
    
    async def similarity_search(self, query_embedding: List[float], user_id: str, match_count: int = 5) -> List[Dict[str, Any]]:
       
        if not self.client:
            print("Supabase client not initialized. Skipping similarity search.")
            return []
        try:
            # Run the blocking RPC call in a separate thread
            result = await asyncio.to_thread(
                self.client.rpc(
                    "match_documents",
                    {"query_embedding": query_embedding, "match_count": match_count, "p_user_id": user_id},
                ).execute
            )
            if result.data:
                return result.data
            else:
                # This can happen if the function exists but returns no rows
                print(f"Similarity search returned no data. Error: {getattr(result, 'error', 'No error reported')}")
                return []
        except Exception as e:
            print(f"An exception occurred during similarity search RPC call: {e}")
            return []