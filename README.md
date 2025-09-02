# LangGraph RAG Agent

A Retrieval-Augmented Generation (RAG) agent built with LangGraph, capable of scraping website content, processing uploaded documents, and answering queries using a combination of vector search and OpenAI LLM fallback. Content is stored in a persistent ChromaDB vector database for fast semantic retrieval.

## Features

- 🌐 **Website Scraping**: Ingest and store website content for retrieval.
- 📄 **Document Ingestion**: Upload and process PDFs, text, HTML, XML, Markdown, CSV, and Excel files.
- 🔍 **Vector Search**: Semantic search over all stored content using OpenAI embeddings.
- 🤖 **RAG Responses**: Generate answers using retrieved content; fallback to OpenAI LLM if no relevant content is found.
- 🗑️ **Content Management**: List and manage available content sources.
- 🧩 **LangGraph Integration**: Designed as a modular graph for use in LangGraph workflows or as a backend API.

## Project Structure

```
acorre-langgraph1/
├── src/agent/
│   ├── graph.py         # Main RAG agent logic (graph, ingestion, query)
│   └── __init__.py      # Exports the graph
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and dependencies
├── langgraph.json       # LangGraph graph configuration
├── uploaded_documents/  # Directory for uploaded documents
├── static/              # Static assets (if any)
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key
- ChromaDB dependencies
- `libmagic` (for document type detection):
  - **Windows**: `pip install python-magic-bin` (may require `pip install msvc-runtime` first)
  - **macOS**: `brew install libmagic`
  - **Linux**: `sudo apt-get install libmagic-dev`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd acorre-langgraph1
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

The main logic is in `src/agent/graph.py`. You can import and use the agent graph, or call the utility functions for ingestion and querying:

### Example: Add Website Content
```python
from agent.graph import add_website_content
add_website_content("https://example.com")
```

### Example: Add Document Content
```python
from agent.graph import add_document_content
add_document_content("./uploaded_documents/sample.pdf")
```

### Example: Query the RAG System
```python
from agent.graph import query_rag
response = query_rag("What is this project about?", website_url="https://example.com")
print(response)
```

### Example: List Content Sources
```python
from agent.graph import get_available_content_sources
sources = get_available_content_sources()
print(sources)
```

## Configuration

- **Chunk size, overlap, and retrieval parameters** can be adjusted in `src/agent/graph.py`.
- **Model, temperature, and API key** are set via environment variables.

## Development

- Lint, format, and test commands are available in the `Makefile`.
- Dependencies are managed in `requirements.txt` and `pyproject.toml`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
