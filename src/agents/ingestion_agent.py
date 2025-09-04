from typing import List, Dict, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ..utils.helpers import generate_collection_name, clean_html_content, get_page_title
from ..utils.storage import VectorStoreManager
from config.settings import settings

class DataIngestionAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        self.storage_manager = VectorStoreManager()
    
    def scrape_website(self, url: str) -> List[Document]:
        """Scrape and process website content"""
        try:
            loader = WebBaseLoader(
                url,
                requests_kwargs={
                    'timeout': settings.REQUEST_TIMEOUT,
                    'headers': {'User-Agent': settings.USER_AGENT}
                }
            )
            documents = loader.load()
            
            page_title = get_page_title(url)
            for doc in documents:
                doc.metadata.update({
                    "source_url": url,
                    "title": page_title,
                    "ingestion_timestamp": "2024-01-01T00:00:00Z"
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error scraping website {url}: {str(e)}")
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks and clean content"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            for chunk in chunks:
                chunk.page_content = clean_html_content(chunk.page_content)
            return chunks
        except Exception as e:
            raise Exception(f"Error processing documents: {str(e)}")
    
    def store_to_vector_db(self, documents: List[Document], 
                           collection_name: Optional[str] = None) -> Chroma:
        """Store documents in Chroma vector database"""
        if not documents:
            raise ValueError("No documents to store")
        
        if not collection_name:
            collection_name = generate_collection_name(documents[0].metadata.get("source_url", ""))
        
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=settings.VECTOR_DB_PATH
            )
            
            self.storage_manager.store_collection_info(
                collection_name=collection_name,
                url=documents[0].metadata.get("source_url", ""),
                document_count=len(documents),
                metadata={
                    "title": documents[0].metadata.get("title", ""),
                    "created_at": "2024-01-01T00:00:00Z"
                }
            )
            
            return vector_store
        except Exception as e:
            raise Exception(f"Error storing to vector DB: {str(e)}")
    
    def ingest_website(self, url: str, collection_name: Optional[str] = None) -> Dict:
        """Full ingestion pipeline"""
        try:
            print(f"🔄 Scraping website: {url}")
            documents = self.scrape_website(url)
            
            print(f"📄 Processing {len(documents)} documents...")
            processed_docs = self.process_documents(documents)
            
            print(f"💾 Storing {len(processed_docs)} chunks to vector DB...")
            vector_store = self.store_to_vector_db(processed_docs, collection_name)
            
            return {
                "success": True,
                "vector_store": vector_store,
                "collection_name": collection_name or generate_collection_name(url),
                "document_count": len(processed_docs),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "vector_store": None,
                "collection_name": None,
                "document_count": 0,
                "error": str(e)
            }
