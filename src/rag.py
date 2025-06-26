import asyncio
import os
import io
from typing import Any, Dict, List
import uuid
import base64
import time
from base64 import b64decode
import aiofiles
from gotrue import Provider
import numpy as np
from PIL import Image
import pydantic
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.models.openai import OpenAIModel
from unstructured.documents.elements import Table, CompositeElement
from pydantic_ai.providers.openai import OpenAIProvider

from unstructured.partition.pdf import partition_pdf
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import json
from datetime import datetime
from load_api import Settings
from agents import OpenAIAgentInit

settings = Settings()

agent_create = OpenAIAgentInit(api_key=settings.OPENAI_API_KEY)

# Supabase configuration
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)


class SupabaseVectorStore:
    """
    Manages interactions with the Supabase vector store, now with non-blocking operations.
    """
    def __init__(self, supabase_client: Client, table_name: str = "documents"):
        self.client = supabase_client
        self.table_name = table_name
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to Supabase vector store asynchronously.
        The blocking 'execute' call is run in a separate thread.
        """
        try:
            batch_size = 20  # Increased batch size slightly
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # --- FIX: Run the blocking database call in a thread ---
                # This prevents the `execute()` call from blocking the event loop.
                result = await asyncio.to_thread(
                    self.client.table(self.table_name).insert(batch).execute
                )
                
                if result.data:
                    print(f"Successfully inserted batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                else:
                    # The result object from the Supabase client might have more error details
                    print(f"Error inserting batch: {getattr(result, 'error', 'Unknown error')}")
                
                # Small delay between batches to avoid overwhelming the DB
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error adding documents to Supabase: {e}")

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using the async method."""
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

# Initialize Supabase vector store
vector_store = SupabaseVectorStore(supabase)

# --- FIX: Create an async wrapper for the blocking doc_partition function ---
async def doc_partition_async(path, file_name, **kwargs):
    """
    Runs the synchronous `partition_pdf` in a separate thread to make it non-blocking.
    """
    # Use functools.partial to pass arguments to the sync function
    from functools import partial
    
    # The blocking function to run
    blocking_func = partial(
        partition_pdf,
        filename=os.path.join(path, file_name),
        **kwargs
    )
    
    loop = asyncio.get_running_loop()
    # `run_in_executor` with None uses the default thread pool executor.
    # This is equivalent to `asyncio.to_thread` (Python 3.9+)
    raw_pdf_elements = await loop.run_in_executor(None, blocking_func)
    return raw_pdf_elements


def data_category(raw_pdf_elements):
    """Categorizes partitioned elements into texts and tables. (This is fast and synchronous)"""
    tables = [str(el) for el in raw_pdf_elements if isinstance(el, Table)]
    texts = [str(el) for el in raw_pdf_elements if isinstance(el, CompositeElement)]
    return {"texts": texts, "tables": tables}

async def encode_image_async(image_path: str) -> str:
    """Encodes an image file into a base64 string asynchronously."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return ""
    try:
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

async def caption_images(image_paths: List[str], image_agent: Agent) -> List[str]:
    """Generates captions for a list of image paths using the pydantic-ai agent."""
    if not image_agent:
        print("Image agent not initialized. Skipping captioning.")
        return [""] * len(image_paths)
        
    print("Capturing images...")
    
    async def caption_single_image(img_path):
        """Helper function to process one image."""
        print(f"  - Captioning image: {os.path.basename(img_path)}")
        base64_image = await encode_image_async(img_path)
        if not base64_image:
            return f"Error encoding image: {os.path.basename(img_path)}"
        try:
            prompt = "Describe this image in detail. What is happening in the image? Be detailed and technical."
            image_data = ImageUrl(url=f"data:image/jpeg;base64,{base64_image}")
            result = await asyncio.wait_for(
                image_agent.run([prompt, image_data]),
                timeout=90.0
            )
            print(f"  - Successfully captioned: {os.path.basename(img_path)}")
            return result.output
        except asyncio.TimeoutError:
            print(f"  - Timeout occurred while captioning image {os.path.basename(img_path)}")
            return f"Timeout error for image: {os.path.basename(img_path)}"
        except Exception as e:
            print(f"  - An error occurred while captioning image {os.path.basename(img_path)}: {e}")
            return f"Error captioning image: {os.path.basename(img_path)} - {str(e)}"

    # Run all captioning tasks concurrently
    tasks = [caption_single_image(path) for path in image_paths]
    captions = await asyncio.gather(*tasks)
    return captions

async def process_and_store_documents(path: str, file_name: str):
    """
    Main async function to process a PDF, generate embeddings, and store in Supabase.
    """
    print("Starting PDF processing...")
    
    try:
        files_before = set(os.listdir(path))
    except FileNotFoundError:
        print(f"Error: The directory '{path}' does not exist. Aborting.")
        return {}

    # Step 1: Process PDF asynchronously.
    raw_pdf_elements = await doc_partition_async(
        path=path,
        file_name=file_name,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        image_output_dir_path=path
    )

    categorized_data = data_category(raw_pdf_elements)
    texts = categorized_data.get("texts", [])
    tables = categorized_data.get("tables", [])
    print(f"Extracted {len(texts)} text chunks and {len(tables)} tables.")

    files_after = set(os.listdir(path))
    new_files = files_after - files_before
    image_paths = [os.path.join(path, f) for f in new_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found and extracted {len(image_paths)} images from the PDF.")

    # Step 2: Generate image captions asynchronously.
    image_captions = []
    if image_paths:
        image_agent = agent_create.create_img_agent(
            sysprompt="You are an expert analyst. Describe the contents of the given image from a technical document in detail."
        )
        image_captions = await caption_images(image_paths, image_agent)

    # Step 3: Prepare all documents for batch insertion.
    documents_to_insert = []
    content_to_embed = []
    
    # Collate all content types that need embedding
    for text in texts:
        content_to_embed.append({"content": text, "type": "text", "metadata": {}})
    for table in tables:
        content_to_embed.append({"content": table, "type": "table", "metadata": {}})
    for i, caption in enumerate(image_captions):
        if caption and not caption.startswith("Error"):
            content_to_embed.append({
                "content": caption,
                "type": "image",
                "metadata": {"image_path": image_paths[i]}
            })

    # Step 4: Generate all embeddings concurrently
    print(f"Generating embeddings for {len(content_to_embed)} content chunks...")
    embedding_tasks = [vector_store.create_embedding(item["content"]) for item in content_to_embed]
    generated_embeddings = await asyncio.gather(*embedding_tasks)
    print("Embeddings generated.")

    # Step 5: Assemble the final documents for insertion
    for i, item in enumerate(content_to_embed):
        embedding = generated_embeddings[i]
        if not embedding: continue  # Skip if embedding failed

        doc = {
            "id": str(uuid.uuid4()),
            "content": item["content"],
            "content_type": item["type"],
            "source_file": file_name,
            "embedding": embedding,
            "metadata": {
                **item["metadata"],
                "created_at": datetime.now().isoformat()
            }
        }
        documents_to_insert.append(doc)

    # Step 6: Insert all documents into Supabase asynchronously.
    if documents_to_insert:
        print(f"Inserting {len(documents_to_insert)} documents into Supabase...")
        await vector_store.add_documents(documents_to_insert)
    else:
        print("No new documents to insert.")

    print("Processing complete!")
    return {
        "total_documents_added": len(documents_to_insert),
        "texts_processed": len(texts),
        "tables_processed": len(tables),
        "images_processed": len(image_captions)
    }

async def main():
    # Make sure the path exists before running
    path = "C:/Users/abhir/Desktop/VexiaSearch/src/"
    file_name = "pdf1.pdf"
    
    if not os.path.exists(os.path.join(path, file_name)):
        print(f"ERROR: The file '{file_name}' was not found in the directory '{path}'")
        print("Please ensure the file exists and the path is correct.")
        return

    try:
        result = await process_and_store_documents(path, file_name)
        print(f"Results: {result}")
    except Exception as e:
        print(f"An error occurred in the main process: {e}", exc_info=True)


if __name__ == "__main__":
    # It's good practice to create the output directory if it doesn't exist
    output_path = "C:/Users/abhir/Desktop/VexiaSearch/src/pdf_test_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
        
    asyncio.run(main())
