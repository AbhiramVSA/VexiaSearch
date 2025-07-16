import asyncio
import os
import io
import uuid
import base64
import time
import aiofiles
import shutil
from typing import Any, Dict, List
from datetime import datetime
from functools import partial
from load_api import Settings

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Pydantic and AI model imports
from agents import OpenAIAgentInit
from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl
from unstructured.documents.elements import Table, Text
from unstructured.partition.pdf import partition_pdf

# Supabase and LangChain imports
from supabase_vector import Client, create_client, SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# Import Pydantic-Settings
from pydantic_settings import BaseSettings, SettingsConfigDict

settings = Settings()
agent_create = OpenAIAgentInit(api_key=settings.OPENAI_API_KEY)

# Database & Embeddings Client Setup 
try:
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
except Exception as e:
    print(f"Unable to initialize Supabase or OpenAI embeddings: {e}")
    print("Please check if the API keys have been set correctly!")
    supabase = None
    embeddings = None

vector_store = SupabaseVectorStore(supabase)

# FastApi and Middleware
app = FastAPI(
    title="Vexia-Search API Endpoint",
    description="Endpoint to upload and process PDF documents for RAG system and process agentic chatting.",
)

# Need to work on deployment, not setup yet
origins = [
    "http://localhost:8080",
    "http://localhost:5173", 
    "http://localhost:3000", 
    "https://vexia-search-ui.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Functions to process uploaded files
async def doc_partition_async(file_path: str, **kwargs):
    
    loop = asyncio.get_running_loop()
    blocking_func = partial(partition_pdf, filename=file_path, **kwargs)
    pdf_elements_raw = await loop.run_in_executor(None, blocking_func)
    return pdf_elements_raw

# To categorise tables seperately.
def data_category(pdf_elements_raw):
   
    tables = []
    texts = []
    for elements in pdf_elements_raw:
        if isinstance(elements, Table):
            tables.append(str(elements))
        elif isinstance(elements, Text): 
            texts.append(str(elements))
            
    return {"texts": texts, "tables": tables}

# Encodes an image file into a base64 string asynchronously.
async def encode_image_async(image_path: str) -> str:
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return "Missing Image"
    try:
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return "Failed to encode Image"

# Generates captions for a list of image paths using the pydantic-ai agent.
async def caption_images(image_paths: List[str], image_agent: Agent) -> List[str]:
    
    if not image_agent:
        print("The image agent was not initialized, skipping the captioning step in the pipeline.")
        return [""] * len(image_paths)

    print("Captioning images...")
    async def caption_single_image(img_path: str) -> str:
        print(f"  - Captioning image: {os.path.basename(img_path)}")
        base64_image = await encode_image_async(img_path)
        if not base64_image:
            return f"Error encoding image: {os.path.basename(img_path)}"
        try:
            prompt = "Describe this image in detail. What is happening in the image? Be detailed and technical."
            image_data = ImageUrl(url=f"data:image/jpeg;base64,{base64_image}")
            result = await asyncio.wait_for(image_agent.run([prompt, image_data]), timeout=90.0)
            print(f"  - Successfully captioned: {os.path.basename(img_path)}")
            return result.output
        except asyncio.TimeoutError:
            print(f"  - Timeout occurred while captioning image {os.path.basename(img_path)}")
            return f"Timeout error for image: {os.path.basename(img_path)}"
        except Exception as e:
            print(f"  - An error occurred while captioning image {os.path.basename(img_path)}: {e}")
            return f"Error captioning image: {os.path.basename(img_path)} - {str(e)}"

    tasks = [caption_single_image(path) for path in image_paths]
    captions = await asyncio.gather(*tasks)
    return captions

# Main async function to process a SINGLE PDF, generate embeddings, and store in Supabase.
async def process_and_store_document(file_path: str, output_path: str, user_id: str):
    
    file_name = os.path.basename(file_path)
    print(f"Starting processing for: {file_name} (User: {user_id})")

    try:
        files_before = set(os.listdir(output_path))
        raw_pdf_elements = await doc_partition_async(
            file_path=file_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            image_output_dir_path=output_path
        )
        files_after = set(os.listdir(output_path))
        
        if not raw_pdf_elements:
            print(f"[{file_name}] WARNING: No content elements were extracted. The PDF might be empty, scanned, or have an unsupported format.")
        else:
            print(f"[{file_name}] Raw element types found: {[type(el).__name__ for el in raw_pdf_elements]}")

        categorized_data = data_category(raw_pdf_elements)
        texts = categorized_data.get("texts", [])
        tables = categorized_data.get("tables", [])
        print(f"[{file_name}] Extracted {len(texts)} text chunks and {len(tables)} tables.")

        new_files = files_after - files_before
        image_paths = [os.path.join(output_path, f) for f in new_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[{file_name}] Found and extracted {len(image_paths)} images.")

        image_captions = []
        if image_paths:
            image_agent = agent_create.create_img_agent(
                sysprompt="You are an expert analyst. Describe the contents of the given image from a technical document in detail."
            )
            image_captions = await caption_images(image_paths, image_agent)

        content_to_embed = []
        for text in texts:
            content_to_embed.append({"content": text, "type": "text", "metadata": {}})
        for table in tables:
            content_to_embed.append({"content": table, "type": "table", "metadata": {}})
        for i, caption in enumerate(image_captions):
            if caption and not caption.startswith("Error"):
                content_to_embed.append({
                    "content": caption,
                    "type": "image",
                    "metadata": {"original_image_name": os.path.basename(image_paths[i])}
                })
        
        if not content_to_embed:
            print(f"[{file_name}] No content to embed. Skipping embedding and database insertion steps.")
            return

        print(f"[{file_name}] Generating embeddings for {len(content_to_embed)} content chunks...")
        embedding_tasks = [vector_store.create_embedding(item["content"]) for item in content_to_embed]
        generated_embeddings = await asyncio.gather(*embedding_tasks)
        print(f"[{file_name}] Embeddings generated.")

        documents_to_insert = []
        for i, item in enumerate(content_to_embed):
            embedding = generated_embeddings[i]
            if not embedding: continue

            documents_to_insert.append({
                "id": str(uuid.uuid4()),
                "content": item["content"],
                "content_type": item["type"],
                "source_file": file_name,
                "user_id": user_id,
                "embedding": embedding,
                "metadata": {
                    **item["metadata"],
                    "created_at": datetime.now().isoformat()
                }
            })

        if documents_to_insert:
            print(f"[{file_name}] Inserting {len(documents_to_insert)} documents into Supabase...")
            await vector_store.add_documents(documents_to_insert)
        else:
            print(f"[{file_name}] No new documents to insert. This might be due to embedding failures.")

        print(f"[{file_name}] Processing complete!")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

# A wrapper to run the processing for all files and clean up afterward.
async def run_processing_in_background(file_paths: List[str], temp_dir: str, user_id: str):

    print(f"Background task started for {len(file_paths)} files.")
    try:
        image_output_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_output_dir, exist_ok=True)
        processing_tasks = [process_and_store_document(fp, image_output_dir, user_id) for fp in file_paths]
        await asyncio.gather(*processing_tasks)
        print("All background processing tasks finished.")
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

TEMP_FILE_DIR = "/tmp"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)

# Endpoints

class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.get("/ping")
def ping_root():
    return {"message": "RAG Processing API is running. POST files to /deploy to start."}

# Accepts multiple PDF files, saves them, and triggers a background task 
# to process them and add their content to the vector store.
@app.post("/deploy", status_code=202) 
async def deploy_documents(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    files: List[UploadFile] = File(..., description="A list of PDF files to process.")
    ):

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_FILE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    saved_file_paths = []
    try:
        for file in files:
            if file.content_type != 'application/pdf':
                shutil.rmtree(session_dir)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
                )
            
            file_path = os.path.join(session_dir, file.filename)
            try:
                async with aiofiles.open(file_path, 'wb') as out_file:
                    content = await file.read()
                    await out_file.write(content)
                saved_file_paths.append(file_path)
            except Exception as e:
                shutil.rmtree(session_dir)
                raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {e}")

        background_tasks.add_task(
            run_processing_in_background,
            saved_file_paths,
            session_dir,
            user_id
        )

        return {
            "success": True,
            "message": f"Processing started for {len(files)} documents in the background.",
            "document_count": len(files),
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# Answers a user's question based on documents they have previously uploaded.
@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    
    print(f"Creating embedding for message: '{request.message}'")
    query_embedding = await vector_store.create_embedding(request.message)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to create embedding for the message.")

    
    print(f"Searching for documents related to the message for user: {request.user_id}")
    matching_docs = await vector_store.similarity_search(
        query_embedding=query_embedding,
        user_id=request.user_id,
        match_count=10  
    )

    context_pieces = [f"Source: {doc.get('source_file', 'N/A')}\nContent: {doc.get('content', '')}" for doc in matching_docs]
    context = "\n---\n".join(context_pieces)
    
    if not matching_docs:
        print("No relevant documents found for the query.")
        return {
            "response": "I could not find any relevant information in your documents to answer that question.",
            "context": "No relevant documents found."
        }
    
    print(f"Found {len(matching_docs)} relevant document chunks.")

    prompt = f"""
    You are an expert assistant. Your task is to answer the user's question based on the context provided below.
    Synthesize the information from the different document chunks to form a coherent, helpful answer.
    If the context does not contain the necessary information to answer the question, state clearly that you cannot answer based on the provided documents.
    Do not use any external knowledge or make up information not present in the context.
    
    CONTEXT:
    ---
    {context}
    ---

    USER'S QUESTION:
    {request.message}
    """
    try:
        chat_agent = agent_create.create_chat_agent(
            sysprompt="You are a helpful assistant that answers questions strictly based on the provided context.",
            model_name="gpt-4o-mini"
        )
        response = await chat_agent.run(prompt)
        
        return {"response": response.output, "context": context}

    except Exception as e:
        error_message = f"An error occurred with the AI agent: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)
