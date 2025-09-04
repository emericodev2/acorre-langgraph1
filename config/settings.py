import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Vector Database
    VECTOR_DB_PATH = "./chroma_db"
    COLLECTION_PREFIX = "web_rag_"
    
    # Text Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    SEARCH_K = 6
    SEARCH_TYPE = "similarity"
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # Web Scraping
    REQUEST_TIMEOUT = 30
    USER_AGENT = "Mozilla/5.0 (compatible; RAG-Agent/1.0)"

settings = Settings()
