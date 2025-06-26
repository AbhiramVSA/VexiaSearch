import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
import json
from load_api import Settings

settings = Settings()
embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)



class SupabaseQueryManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client = create_client(supabase_url, supabase_key)
        self.embeddings = OpenAIEmbeddings()
        self.table_name = "documents"
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = 0.7,
        content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector database
        
        Args:
            query: The search query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            content_type: Filter by content type ('text', 'table', 'image')
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Call the similarity search function
            result = self.client.rpc(
                'search_similar_documents',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': similarity_threshold,
                    'match_count': k
                }
            ).execute()
            
            documents = result.data if result.data else []
            
            # Filter by content type if specified
            if content_type:
                documents = [doc for doc in documents if doc['content_type'] == content_type]
            
            return documents
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def get_documents_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source file"""
        try:
            result = self.client.table(self.table_name)\
                .select("*")\
                .eq("source_file", source_file)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error retrieving documents by source: {e}")
            return []
    
    def get_documents_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """Get all documents of a specific type"""
        try:
            result = self.client.table(self.table_name)\
                .select("*")\
                .eq("content_type", content_type)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error retrieving documents by type: {e}")
            return []
    
    def delete_documents_by_source(self, source_file: str) -> bool:
        """Delete all documents from a specific source file"""
        try:
            result = self.client.table(self.table_name)\
                .delete()\
                .eq("source_file", source_file)\
                .execute()
            
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    async def hybrid_search(
        self, 
        query: str, 
        k: int = 10,
        similarity_threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform hybrid search across all content types
        Returns results grouped by content type
        """
        results = {
            'text': [],
            'table': [],
            'image': []
        }
        
        # Search each content type
        for content_type in ['text', 'table', 'image']:
            type_results = await self.similarity_search(
                query=query,
                k=k//3,  # Distribute results across types
                similarity_threshold=similarity_threshold,
                content_type=content_type
            )
            results[content_type] = type_results
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            # Count total documents
            total_result = self.client.table(self.table_name)\
                .select("*", count="exact")\
                .execute()
            
            total_count = total_result.count if total_result.count else 0
            
            # Count by type
            type_counts = {}
            for content_type in ['text', 'table', 'image']:
                type_result = self.client.table(self.table_name)\
                    .select("*", count="exact")\
                    .eq("content_type", content_type)\
                    .execute()
                
                type_counts[content_type] = type_result.count if type_result.count else 0
            
            # Get unique source files
            sources_result = self.client.table(self.table_name)\
                .select("source_file")\
                .execute()
            
            unique_sources = list(set([doc['source_file'] for doc in sources_result.data])) if sources_result.data else []
            
            return {
                'total_documents': total_count,
                'by_type': type_counts,
                'unique_sources': len(unique_sources),
                'source_files': unique_sources
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

# Example usage
async def example_usage():
    # Initialize the query manager
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    query_manager = SupabaseQueryManager(SUPABASE_URL, SUPABASE_KEY)
    
    # Example 1: Simple similarity search
    results = await query_manager.similarity_search(
        query="what is wilcoxn",
        k=5,
        similarity_threshold=0.7
    )
    print(f"Found {len(results)} similar documents")
    
    # Example 2: Search only tables
    table_results = await query_manager.similarity_search(
        query="what is wilcoxn",
        k=3,
        content_type="table"
    )
    print(f"Found {len(table_results)} similar tables")
    
    # Example 3: Hybrid search
    hybrid_results = await query_manager.hybrid_search(
        query="what is wilcoxn",
        k=9
    )
    print(f"Hybrid search results:")
    for content_type, docs in hybrid_results.items():
        print(f"  {content_type}: {len(docs)} documents")
    
    # Example 4: Get database statistics
    stats = query_manager.get_database_stats()
    print(f"Database stats: {stats}")

# Run the examples
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())