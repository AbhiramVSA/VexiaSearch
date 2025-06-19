# Vexia Search

[](https://www.google.com/search?q=https://github.com/your-username/CortexEngine/actions/workflows/ci.yml)
[](https://opensource.org/licenses/MIT)

A cognitive search and analytics engine designed to perform semantic retrieval and generate deep insights from large document sets using a multi-agent RAG architecture.

## 核心功能 (Core Features)

  - 🚀 **Scalable Data Ingestion**: Asynchronous ingestion pipeline built with FastAPI and Celery to handle high-volume data streams.
  - 🧠 **Cognitive Search**: Leverages Retrieval-Augmented Generation (RAG) with vector embeddings for context-aware, semantic search, not just keyword matching.
  - 🤖 **Multi-Agent Analytics**: Deploys a team of specialized AI agents to analyze, summarize, and extract key themes from search results.
  - 🐳 **Fully Containerized**: All services are containerized with Docker for consistent, reproducible deployments and scalability.
  - ⚙️ **CI/CD Ready**: Integrated with GitHub Actions for automated testing and build processes.

## Architecture Overview

The system is designed as a distributed set of microservices, ensuring separation of concerns and scalability. The primary components are the Ingestion API, the Celery Task Queue for background processing, the Core Search API, and the Multi-Agent Analytics Engine.

*(A clear architectural diagram is crucial for demonstrating sophisticated design)*

## Technology Stack

| Component             | Technology                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Backend** | Python, FastAPI, Celery                                                                                     |
| **AI / ML** | RAG, Multi-Agent Systems, LLMs (GROQ), Sentence-Transformers, Pydantic-AI                                     |
| **Databases** | Weaviate (Vector DB), PostgreSQL (Metadata), RabbitMQ (Message Broker)                                      |
| **DevOps & CI/CD** | Docker, Docker Compose, GitHub Actions, Git                                                                 |
| **Frameworks/Libs** | LangChain, FAISS                                                                                            |

## Getting Started

### Prerequisites

  - Docker & Docker Compose
  - Python 3.10+
  - API Keys

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/CortexEngine.git
    cd VexiaSearch
    ```

2.  **Configure environment variables:**
    Create a `.env` file from the example and populate it with your credentials.

    ```bash
    cp .env.example .env
    ```

    ```ini
    # .env
    
    ```

3.  **Build and launch the containers:**
    This command will build the images and start all the services (API, worker, database, message queue).

    ```bash
    docker-compose up --build -d
    ```

The API will be accessible at `http://localhost:8000/docs`.

## Usage

Interact with the engine via the RESTful API endpoints.

### 1\. Ingest a Document

Send a document to the ingestion pipeline. The system will process it asynchronously.


```

### 2\. Perform a Cognitive Search

Ask a natural language question. The engine will retrieve relevant context and generate an answer.



**API Response:**


### 3\. Run Multi-Agent Analysis

Trigger the analytics engine to get deeper insights from the documents relevant to your query.



## Showcase

