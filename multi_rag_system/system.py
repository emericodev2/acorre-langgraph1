# multi_rag_system/system.py
"""Core Multi-RAG system implementation."""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

# Core LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiRAGSystem:
    """Multi-source RAG system for LangGraph Platform."""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 similarity_threshold: float = None):
        
        # Get configuration from environment or use defaults
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        self.similarity_threshold = similarity_threshold or float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # Initialize OpenAI components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use more cost-effective model for platform
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Vector stores
        self.website_vectorstore = None
        self.document_vectorstore = None
        
        logger.info(f"MultiRAGSystem initialized with chunk_size={self.chunk_size}, similarity_threshold={self.similarity_threshold}")
        
    async def ingest_websites(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest and process websites for RAG."""
        logger.info(f"Ingesting {len(urls)} websites...")
        
        documents = []
        successful_urls = []
        failed_urls = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update({
                        "source_type": "website",
                        "url": url,
                        "ingested_at": datetime.now().isoformat()
                    })
                
                documents.extend(docs)
                successful_urls.append(url)
                logger.info(f"Successfully loaded: {url}")
                
            except Exception as e:
                logger.error(f"Failed to load {url}: {str(e)}")
                failed_urls.append({"url": url, "error": str(e)})
        
        if documents:
            splits = self.text_splitter.split_documents(documents)
            self.website_vectorstore = FAISS.from_documents(splits, self.embeddings)
            logger.info(f"Created website vector store with {len(splits)} chunks")
        
        return {
            "total_attempted": len(urls),
            "successful": len(successful_urls),
            "failed": len(failed_urls),
            "successful_urls": successful_urls,
            "failed_urls": failed_urls,
            "total_chunks": len(splits) if documents else 0
        }
    
    async def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest and process documents for RAG."""
        logger.info(f"Ingesting {len(file_paths)} documents...")
        
        documents = []
        successful_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                mime_type, _ = mimetypes.guess_type(file_path)
                
                if mime_type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    loader = Docx2txtLoader(file_path)
                elif mime_type and mime_type.startswith("text/"):
                    loader = TextLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    failed_files.append({"file": file_path, "error": "Unsupported file type"})
                    continue
                
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update({
                        "source_type": "document",
                        "file_path": file_path,
                        "ingested_at": datetime.now().isoformat()
                    })
                
                documents.extend(docs)
                successful_files.append(file_path)
                logger.info(f"Successfully loaded: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                failed_files.append({"file": file_path, "error": str(e)})
        
        if documents:
            splits = self.text_splitter.split_documents(documents)
            self.document_vectorstore = FAISS.from_documents(splits, self.embeddings)
            logger.info(f"Created document vector store with {len(splits)} chunks")
        
        return {
            "total_attempted": len(file_paths),
            "successful": len(successful_files),
            "failed": len(failed_files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": len(splits) if documents else 0
        }
    
    async def search_websites(self, query: str, k: int = 5) -> tuple[List[Document], float]:
        """Search website vector store."""
        if not self.website_vectorstore:
            return [], 0.0
        
        try:
            results = self.website_vectorstore.similarity_search_with_score(query, k=k)
            relevant_docs = [
                doc for doc, score in results 
                if score <= (1.0 - self.similarity_threshold)
            ]
            
            if relevant_docs:
                avg_score = sum(score for _, score in results[:len(relevant_docs)]) / len(relevant_docs)
                confidence = max(0.0, 1.0 - avg_score)
            else:
                confidence = 0.0
            
            return relevant_docs, confidence
            
        except Exception as e:
            logger.error(f"Website search error: {str(e)}")
            return [], 0.0
    
    async def search_documents(self, query: str, k: int = 5) -> tuple[List[Document], float]:
        """Search document vector store."""
        if not self.document_vectorstore:
            return [], 0.0
        
        try:
            results = self.document_vectorstore.similarity_search_with_score(query, k=k)
            relevant_docs = [
                doc for doc, score in results 
                if score <= (1.0 - self.similarity_threshold)
            ]
            
            if relevant_docs:
                avg_score = sum(score for _, score in results[:len(relevant_docs)]) / len(relevant_docs)
                confidence = max(0.0, 1.0 - avg_score)
            else:
                confidence = 0.0
            
            return relevant_docs, confidence
            
        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return [], 0.0
    
    async def llm_search(self, query: str) -> tuple[str, float]:
        """Perform LLM search as fallback."""
        try:
            search_prompt = ChatPromptTemplate.from_template("""
            The user asked: "{query}"
            
            The RAG systems could not find sufficient information to answer this query.
            Please provide a comprehensive answer based on your knowledge.
            Be thorough and informative. If you're unsure, indicate that clearly.
            
            Query: {query}
            """)
            
            chain = search_prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({"query": query})
            
            return result, 0.6  # Medium confidence for LLM fallback
            
        except Exception as e:
            logger.error(f"LLM search error: {str(e)}")
            return f"Unable to perform search: {str(e)}", 0.0
    
    async def synthesize_answer(self, query: str, website_results: List[Document], 
                              document_results: List[Document], llm_result: str = None) -> str:
        """Synthesize final answer from all sources."""
        
        website_context = "\n\n".join([doc.page_content for doc in website_results]) if website_results else ""
        document_context = "\n\n".join([doc.page_content for doc in document_results]) if document_results else ""
        
        synthesis_prompt = ChatPromptTemplate.from_template("""
        Based on the following information sources, provide a comprehensive answer to the user's query.
        
        Query: {query}
        
        Website Information:
        {website_context}
        
        Document Information:
        {document_context}
        
        Additional Information:
        {llm_context}
        
        Instructions:
        1. Synthesize information from all available sources
        2. If sources conflict, acknowledge the differences
        3. Clearly indicate source types when relevant
        4. If no relevant information was found, state that clearly
        5. Provide a clear, well-structured answer
        
        Answer:
        """)
        
        try:
            chain = synthesis_prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({
                "query": query,
                "website_context": website_context or "No relevant website information found.",
                "document_context": document_context or "No relevant document information found.",
                "llm_context": llm_result or "No additional LLM search performed."
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            return f"Unable to synthesize answer: {str(e)}"
