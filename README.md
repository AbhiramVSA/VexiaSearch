# Vexia Search

Vexia Search is a cognitive search and analytics engine designed to perform semantic retrieval and generate deep insights from large PDF document sets using a multi-agent RAG architecture. The system leverages vector embeddings for context-aware search and provides a conversational interface for document interaction.

## Architecture & Data Flow

The system is a monolithic FastAPI application designed for serverless deployment on platforms like Vercel. It operates via two primary asynchronous workflows:

**1. Ingestion Flow (`/deploy` endpoint):**

  - A user uploads one or more PDF files via a `POST` request.
  - FastAPI receives the files and saves them to a temporary directory (`/tmp` for Vercel compatibility).
  - A `BackgroundTask` is initiated to process each PDF without blocking the server.
  - The `unstructured` library partitions the PDF, extracting text, tables, and identifying images.
  - Text and table content is chunked and then embedded using OpenAI's embedding models via `langchain-openai`.
  - The content, metadata (`user_id`, `source_file`), and its corresponding vector embedding are stored in a Supabase (PostgreSQL) table equipped with the `pgvector` extension.

**2. Retrieval & Chat Flow (`/chat` endpoint):**

  - A user submits a natural language question and `user_id` via a `POST` request.
  - The user's question is converted into a vector embedding.
  - A similarity search is executed against the Supabase vector store by calling a custom `match_documents` RPC function. This function filters for high-quality context by using a similarity threshold.
  - The top-matching document chunks are retrieved and compiled into a context block.
  - This context, along with the original question, is formatted into a detailed prompt.
  - The prompt is sent to a `pydantic-ai` agent powered by an OpenAI model (`gpt-4o-mini`) to generate a context-aware answer.
  - The final response is returned to the user.

## Technology Stack

| Component           | Technology / Library                                                                |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Backend** | Python, FastAPI, Uvicorn                                                            |
| **AI / RAG** | `pydantic-ai`, `langchain-openai`, `unstructured`                                     |
| **Database** | Supabase (PostgreSQL with pgvector)                                                 |
| **Deployment** | Vercel                                                                              |
| **Core Libraries** | `asyncio`, `aiofiles`, `pydantic-settings`                                          |

## Local Development Setup

### Prerequisites

  - Python 3.11+
  - A Supabase project with the `vector` extension enabled.
  - An OpenAI API Key.

### 1\. Clone the Repository

```bash
git clone https://github.com/AbhiramVSA/VexiaSearch.git
cd VexiaSearch
```

### 2\. Set Up Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate
```

### 3\. Install Dependencies

Install the required packages from the curated `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

Create a `.env` file in the root directory and populate it with your credentials.

```ini
# .env
OPENAI_API_KEY="sk-..."
SUPABASE_URL="https://your-project-ref.supabase.co"
SUPABASE_KEY="your-supabase-anon-key"
```

### 5\. Set Up Supabase Database Function

For the chat functionality to work, you must run the following SQL query in your Supabase project's SQL Editor. This creates the function that performs the vector similarity search.

```sql
-- Create a function to search for documents based on user_id, a similarity threshold, and a match count.
create or replace function match_documents (
  query_embedding vector(1536),
  match_threshold float, 
  match_count int,
  p_user_id text
)
returns table (
  id uuid,
  content text,
  content_type text,
  source_file text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.content_type,
    documents.source_file,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.user_id = p_user_id
    and 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

Also ensure your `documents` table has a `user_id` text column.

### 6\. Run the Application

Use `uvicorn` to run the FastAPI server.

```bash
uvicorn src.rag:app --reload
```

The API documentation will be available at `http://12.0.0.1:8000/docs`.

## API Endpoints & Usage

### Deploy Documents

  - **Endpoint**: `/deploy`
  - **Method**: `POST`
  - **Description**: Uploads one or more PDF files for processing and storage. This is an asynchronous operation that runs in the background.

**Example (PowerShell):**

```powershell
curl -X POST http://127.0.0.1:8000/deploy `
  -F "user_id=user_123" `
  -F "files=@C:\path\to\your\document1.pdf" `
  -F "files=@C:\path\to\your\document2.pdf"
```

### Chat with Documents

  - **Endpoint**: `/chat`
  - **Method**: `POST`
  - **Description**: Ask a question about the documents you have uploaded for a specific `user_id`.

**Example (PowerShell):**

```powershell
# Step 1: Create a PowerShell object with your data
$body = @{
    "message" = "What are the key skills listed in the resume?"
    "user_id" = "user_123"
}

# Step 2: Send the request using Invoke-RestMethod
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/chat' -Method 'POST' -ContentType 'application/json' -Body ($body | ConvertTo-Json)
```

## Deployment to Vercel

This application is configured for deployment on Vercel.

1.  **Configuration Files**: Ensure you have a `vercel.json` and a minimal `requirements.txt` in your project's root directory as previously discussed.
2.  **Temporary Directory**: The code has been configured to use the `/tmp` directory for file uploads, which is compatible with Vercel's read-only filesystem.
3.  **Environment Variables**: Set your `OPENAI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` in the Vercel project settings dashboard.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
