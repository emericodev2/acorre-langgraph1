# LangGraph RAG Project with Pinecone

This project implements a RAG (Retrieval-Augmented Generation) system using LangGraph, OpenAI, and Pinecone. It consists of two main workflows:
- **Ingest Graph**: Loads documents from URLs or files and stores them in Pinecone
- **Query Graph**: Retrieves relevant documents and generates answers using OpenAI

## Prerequisites

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com)
2. **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io)
3. **LangGraph Cloud Account**: Sign up at [LangGraph](https://langsmith.langchain.com)

## Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=universal-rag-index
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the graphs locally:
```bash
# Start development server (serves both graphs)
langgraph dev

# This will start a server at http://localhost:8123
# You can then access:
# - http://localhost:8123/ingest for the ingestion graph
# - http://localhost:8123/query for the query graph
```

3. Alternative: Test specific graphs using Python:
```bash
# Test ingestion graph
python -c "
from src.ingest_graph import graph_ingest
result = graph_ingest.invoke({
    'url': 'https://example.com',
    'file_path': None,
    'source_type': '',
    'source_name': '',
    'docs': [],
    'chunks': []
})
print(result)
"

# Test query graph
python -c "
from src.query_graph import graph_query
result = graph_query.invoke({
    'query': 'What is this about?',
    'source_type': None,
    'source_name': None,
    'answer': ''
})
print(result)
"
```

## Deployment to LangGraph Cloud

### Step 1: Install LangGraph CLI
```bash
pip install langgraph-cli
```

### Step 2: Login to LangGraph
```bash
langgraph login
```

### Step 3: Deploy the Project
```bash
# Deploy both graphs
langgraph deploy

# Or deploy with a specific name
langgraph deploy --name my-rag-app
```

### Step 4: Set Environment Variables in LangGraph Cloud

After deployment, set the following environment variables in the LangGraph Cloud dashboard:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `INDEX_NAME` (optional, defaults to "universal-rag-index")

## Testing the Deployed Application

Once deployed, you can test using curl or any HTTP client:

### Ingest a Website
```bash
curl -X POST "https://your-deployment-url/ingest/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "url": "https://example.com/article",
      "file_path": null,
      "source_type": "",
      "source_name": "",
      "docs": [],
      "chunks": []
    }
  }'
```

### Query the Knowledge Base
```bash
curl -X POST "https://your-deployment-url/query/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "query": "What are the main topics discussed?",
      "source_type": null,
      "source_name": null,
      "answer": ""
    }
  }'
```

## Usage Examples

### Ingesting a Website
```json
{
  "url": "https://example.com/article",
  "file_path": null,
  "source_type": "",
  "source_name": "",
  "docs": [],
  "chunks": []
}
```

### Ingesting a PDF File
```json
{
  "url": null,
  "file_path": "/path/to/document.pdf",
  "source_type": "",
  "source_name": "",
  "docs": [],
  "chunks": []
}
```

### Querying the Knowledge Base
```json
{
  "query": "What are the main topics discussed?",
  "source_type": null,
  "source_name": null,
  "answer": ""
}
```

### Filtered Querying
```json
{
  "query": "What are the main topics discussed?",
  "source_type": "website",
  "source_name": "https://example.com/article",
  "answer": ""
}
```

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and environment variables
│   ├── ingest_graph.py        # Document ingestion workflow
│   ├── query_graph.py         # Query and retrieval workflow
│   └── main.py                # Graph exports for CLI
├── langgraph.json             # LangGraph configuration
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
└── .gitignore                # Git ignore rules
```

## LangGraph CLI Commands

### Development
```bash
langgraph dev                  # Start development server
langgraph dev --port 8124     # Start on different port
```

### Deployment
```bash
langgraph deploy              # Deploy to LangGraph Cloud
langgraph deploy --name app   # Deploy with specific name
langgraph list               # List deployments
langgraph logs               # View deployment logs
```

### Testing
```bash
# The development server provides a web interface at:
# http://localhost:8123
# 
# You can test graphs interactively through the web UI
# or send HTTP requests to:
# - POST http://localhost:8123/ingest/invoke
# - POST http://localhost:8123/query/invoke
```

## Key Features

- **Multi-source ingestion**: Supports websites, PDFs, and other document formats
- **Metadata filtering**: Query specific sources or document types
- **Chunking strategy**: Recursive text splitting with overlap for better retrieval
- **Vector storage**: Pinecone serverless for scalable vector search
- **LLM integration**: OpenAI GPT-4o-mini for answer generation

## Troubleshooting

### Common Issues

1. **"No dependencies found in config"**: Make sure `langgraph.json` has the `dependencies` array
2. **API Key Errors**: Ensure all environment variables are set in `.env` file
3. **Pinecone Index Issues**: The system automatically creates the index if it doesn't exist
4. **Import Errors**: Make sure you're running from the project root directory

### Debugging

Enable debug logging by setting:
```bash
export LANGCHAIN_DEBUG=true
```

Check LangGraph logs:
```bash
langgraph logs
```

## Next Steps

- Add support for more document types
- Implement conversation memory
- Add document summarization
- Create a web interface
- Add monitoring and analytics