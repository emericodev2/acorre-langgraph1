<!-- README_PLATFORM.md - Platform-specific deployment guide -->
# Multi-RAG System - LangGraph Platform Deployment

This guide covers deploying the Multi-RAG system specifically on the LangGraph Platform.

## Pre-deployment Setup

1. **Project Structure**
```
multi-rag-system/
├── multi_rag_system/
│   ├── __init__.py
│   ├── graph.py          # Main graph definition
│   ├── system.py         # Core RAG system
│   ├── state.py          # State definitions
│   ├── nodes.py          # Node implementations
│   └── routing.py        # Routing logic
├── langgraph.json        # Platform configuration
├── pyproject.toml        # Package configuration
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
└── deployment_platform.py # Platform utilities
```

2. **Environment Variables**
Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7
DEFAULT_WEBSITES=https://docs.python.org/3/,https://langchain-ai.github.io/langgraph/
```

## Deployment Commands

### 1. Install LangGraph CLI
```bash
pip install langgraph-cli
```

### 2. Login to Platform
```bash
langgraph auth login
```

### 3. Deploy
```bash
# Deploy with automatic configuration
langgraph up

# Or deploy with specific configuration
langgraph deploy --config langgraph.json
```

### 4. Test Deployment
```bash
# Test the deployed graph
langgraph invoke multi_rag_agent '{"query": "What is Python?"}'
```

## Platform-Specific Features

### 1. Automatic Initialization
The system automatically ingests default websites on startup if `DEFAULT_WEBSITES` is configured.

### 2. Health Monitoring
Built-in health check endpoint for platform monitoring:
```python
health_status = await deployment.health_check()
```

### 3. Content Management
Platform-compatible content ingestion:
```python
await deployment.ingest_content({
    "websites": ["https://example.com"],
    "documents": ["/path/to/doc.pdf"]
})
```

## Usage on Platform

### 1. Direct Graph Invocation
```python
from langgraph import invoke_graph

result = await invoke_graph(
    "multi_rag_agent",
    {
        "query": "Your question here",
        "max_tokens": 4000,
        "temperature": 0.1
    }
)
```

### 2. Platform API
If platform exposes REST API:
```bash
curl -X POST "https://your-deployment.langgraph.com/invoke" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is LangGraph?"}'
```

### 3. Streaming Support
```python
async for chunk in stream_graph("multi_rag_agent", {"query": "Tell me about AI"}):
    print(chunk)
```

## Environment Configuration

### Required Variables
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional Variables
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200) 
- `SIMILARITY_THRESHOLD`: Search threshold (default: 0.7)
- `DEFAULT_WEBSITES`: Comma-separated URLs to ingest on startup

## Monitoring and Debugging

### 1. Platform Logs
Check platform logs for system messages:
```bash
langgraph logs multi_rag_agent
```

### 2. Health Checks
Monitor system health:
```python
health = await deployment.health_check()
print(f"Status: {health['status']}")
print(f"Ready: {health['system_ready']}")
```

### 3. Performance Metrics
Track query performance through platform metrics dashboard.

## Scaling and Optimization

### 1. Resource Configuration
Adjust platform resources based on usage:
- Memory: Increase for larger vector stores
- CPU: Scale for concurrent queries
- Storage: Ensure sufficient space for documents

### 2. Caching Strategy
The system uses FAISS for efficient vector storage and retrieval.

### 3. Cost Optimization
- Use `gpt-4o-mini` for cost-effective LLM fallback
- Implement query result caching
- Optimize chunk sizes for your use case

## Security Considerations

1. **API Keys**: Store securely in platform environment variables
2. **Content Validation**: Validate URLs and documents before ingestion
3. **Rate Limiting**: Platform handles request rate limiting
4. **Access Control**: Configure platform access permissions

## Troubleshooting

### Common Issues

1. **Deployment Fails**
   - Check `langgraph.json` syntax
   - Verify all dependencies in `requirements.txt`
   - Ensure environment variables are set

2. **Graph Not Found**
   - Verify graph