import os
from dotenv import load_dotenv

# Load env vars locally (LangGraph Cloud injects them automatically)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "universal-rag-index")
