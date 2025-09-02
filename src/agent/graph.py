import os
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
import chromadb
import filetype # Replaced magic for file type detection
from urllib.parse import urlparse, urlunparse # For URL canonicalization
import ntpath # For path normalization on Windows
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import RateLimitError, AuthenticationError
import traceback


# Load environment variables
load_dotenv()

# Define UPLOAD_DIR for consistent path normalization within graph.py
UPLOAD_DIR = "./uploaded_documents"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    website_url: str | None
    document_path: str | None # New: path to document to be processed
    rag_results: List[Document]
    should_reroute_to_llm: bool # New flag for rerouting after RAG attempt

# Initialize components
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

api_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY: {api_key}")

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    client=chroma_client,
    embedding_function=embeddings,
    collection_name="website_content"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Helper function to canonicalize URLs for consistent storage and retrieval
def _canonicalize_url(url: str) -> str:
    """Standardizes a URL to ensure consistent storage and retrieval in ChromaDB."""
    parsed_url = urlparse(url)
    # Ensure scheme is https (or http if that's what WebBaseLoader commonly uses and we want to stick to it)
    # For consistency, let's enforce https if not explicitly http
    scheme = parsed_url.scheme if parsed_url.scheme else "https"
    if scheme == "http" and "https" not in url:
        pass # Keep http if explicitly provided
    else:
        scheme = "https"
    
    # Remove trailing slash unless it's the root path (e.g., 'https://example.com/' becomes 'https://example.com')
    # or if it's explicitly part of the path (e.g., 'https://example.com/path/')
    path = parsed_url.path
    if path.endswith('/') and len(path) > 1: # Only remove if not just '/' for root
        path = path.rstrip('/')

    # Reconstruct the URL
    canonical_url = urlunparse(parsed_url._replace(scheme=scheme, path=path))
    return canonical_url

# Helper function to normalize document paths for consistent storage and retrieval
def _normalize_document_path(file_path: str) -> str:
    """Standardizes a document file path for consistent storage and retrieval."""
    # Replace backslashes with forward slashes for consistency across OS
    normalized_path = file_path.replace('\\', '/')

    # Define the expected base for document paths
    expected_base = UPLOAD_DIR.replace('\\', '/') # Ensure base also uses forward slashes
    
    # If the path is absolute and within the expected upload directory, make it relative
    if os.path.isabs(normalized_path) and normalized_path.startswith(os.path.abspath(UPLOAD_DIR).replace('\\', '/')):
        normalized_path = os.path.relpath(normalized_path, start=UPLOAD_DIR).replace('\\', '/')
        
    # Ensure the path consistently starts with './uploaded_documents/'
    if not normalized_path.startswith(expected_base):
        if normalized_path.startswith(expected_base.lstrip('./')):
            # Path like 'uploaded_documents/file.txt', prepend '.'
            normalized_path = f'./{normalized_path}'
        else:
            # Path is just a filename, prepend full expected_base
            normalized_path = f'{expected_base}/{normalized_path}'
    
    # Clean up any redundant slashes and ensure consistent format
    normalized_path = os.path.normpath(normalized_path).replace('\\', '/')
    
    # Remove any duplicate './' if it was added unnecessarily, while preserving a single leading './'
    if normalized_path.startswith('././'):
        normalized_path = normalized_path[2:] # Remove one of the './'

    return normalized_path

def scrape_website(state: State) -> State:
    """Scrape website content and store in vector database"""
    try:
        url = state["website_url"]
        if not url:
            print("No website_url provided for scraping.")
            return state

        canonical_url = _canonicalize_url(url)
        print(f"DEBUG: Canonicalized URL for scraping: {canonical_url}")
        
        # Check if we already have content for this URL
        existing_docs = vectorstore.similarity_search(
            canonical_url, # Search using canonical URL
            k=1, 
            filter={"source": canonical_url} # Filter using canonical URL
        )
        
        if existing_docs:
            # Content already exists, skip scraping
            print(f"Content for {canonical_url} already exists, skipping scrape.")
            return state
        
        # Scrape the website
        loader = WebBaseLoader(canonical_url) # Use canonical URL for loader
        documents = loader.load()
        
        # Split documents into chunks
        splits = text_splitter.split_documents(documents)
        
        # Add metadata
        for split in splits:
            split.metadata["source"] = canonical_url # Store canonical URL
            split.metadata["type"] = "website_content"
            print(f"DEBUG: Adding website chunk with source metadata: {split.metadata['source']}")  # Debug log
        
        # Try storing in vectorstore with primary (OpenAI) embeddings
        try:
            vectorstore.add_documents(splits)
            print(f"Successfully scraped and stored content from {canonical_url} (OpenAI embeddings).")

        except (RateLimitError, AuthenticationError, Exception) as e:
            # If OpenAI quota or auth issue, fallback to HuggingFace embeddings
            print(f"OpenAI embeddings failed: {e}")
            traceback.print_exc()

            print("Falling back to HuggingFace embeddings...")
            hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            fallback_vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=hf_embeddings
            )
            fallback_vectorstore.add_documents(splits)
            print(f"Successfully scraped and stored content from {canonical_url} (HuggingFace embeddings).")

        
    except Exception as e:
        print(f"Error scraping website: {e}")
    
    return state

def search_rag_content(state: State) -> State:
    """Search for relevant content in the RAG database"""
    try:
        query = state["query"]
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=5)
        
        if docs:
            state["rag_results"] = docs
            print(f"Found {len(docs)} documents for RAG search.")
        else:
            state["rag_results"] = []
            print("No documents found for RAG search.")
            
        state["should_reroute_to_llm"] = False # Reset for this path

    except Exception as e:
        print(f"Error searching RAG content: {e}")
        state["rag_results"] = []
        state["should_reroute_to_llm"] = True # Force fallback on search error
    
    return state

def process_document(state: State) -> State:
    """Process a document, load, split, and store in vector database"""
    if not state.get("document_path"):
        print("No document_path provided for processing.")
        return state

    doc_path = state["document_path"]
    print(f"Processing document: {doc_path}")

    try:
        # Normalize the document path for consistent storage
        canonical_doc_path = _normalize_document_path(doc_path)
        print(f"DEBUG: Normalized document path for processing and storage: {canonical_doc_path}")

        # Determine loader based on file type (prioritize extension for common types)
        file_extension = os.path.splitext(doc_path)[1].lower()
        mime_type = None
        kind = filetype.guess(doc_path)

        if file_extension == ".csv":
            mime_type = "text/csv"
            print(f"DEBUG: Detected MIME type from extension: {mime_type} for {doc_path}")
        elif file_extension == ".xlsx" or file_extension == ".xls":
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            print(f"DEBUG: Detected MIME type from extension: {mime_type} for {doc_path}")
        elif kind:
            mime_type = kind.mime
            print(f"DEBUG: Detected MIME type from filetype.guess(): {mime_type} (extension: {kind.extension}) for {doc_path}") # Refined Debug log
        else:
            print(f"DEBUG: Could not determine file type for {doc_path}. Attempting with UnstructuredFileLoader (fallback).")

        loader = None
        if mime_type and "pdf" in mime_type:
            loader = PyPDFLoader(doc_path)
            print(f"DEBUG: Using PyPDFLoader for {doc_path}")
        elif mime_type and (
            "text" in mime_type or 
            "xml" in mime_type or 
            "html" in mime_type or 
            "markdown" in mime_type or
            "csv" in mime_type or 
            "spreadsheetml" in mime_type or 
            "excel" in mime_type
        ):
            loader = UnstructuredFileLoader(doc_path)
            print(f"DEBUG: Using UnstructuredFileLoader for {doc_path} (MIME: {mime_type})")
        else:
            print(f"DEBUG: Attempting to load unknown/unhandled type ({mime_type}) with UnstructuredFileLoader (general fallback).")
            loader = UnstructuredFileLoader(doc_path)

        if not loader:
            print(f"ERROR: No suitable loader found for {mime_type} for document {doc_path}.")
            return state

        documents = loader.load()
        print(f"DEBUG: Loader loaded {len(documents)} documents from {doc_path}")
        
        if not documents:
            print(f"WARNING: No content loaded by loader for {doc_path}. Skipping splitting and embedding.")
            return state

        splits = text_splitter.split_documents(documents)
        print(f"DEBUG: Text splitter created {len(splits)} splits from {doc_path}")
        
        if not splits:
            print(f"WARNING: No splits created from {doc_path}. Skipping embedding.")
            return state

        # Add metadata for source and type
        for i, split in enumerate(splits):
            split.metadata["source"] = canonical_doc_path # Store normalized document path
            split.metadata["type"] = "document_content"
            if i < 3: # Log first 3 splits metadata for brevity
                print(f"DEBUG: Split {i} metadata['source']: {split.metadata['source']}")
        
        try:
            vectorstore.add_documents(splits)
            print(f"DEBUG: Successfully added {len(splits)} splits to vectorstore for {canonical_doc_path}")
        except Exception as vector_e:
            print(f"ERROR: Failed to add documents to vectorstore for {canonical_doc_path}: {vector_e}")
            raise vector_e # Re-raise to ensure upstream catches it
        
        print(f"Successfully processed and stored document: {canonical_doc_path}")

    except Exception as e:
        print(f"Error processing document {doc_path}: {e}")
    finally:
        # Clean up the uploaded file after processing
        # NOTE: 'doc_path' is the temporary location, 'canonical_doc_path' is how it's stored in DB.
        if os.path.exists(doc_path):
            os.remove(doc_path)
            print(f"Cleaned up temporary uploaded document: {doc_path}")
    
    return state

def rag_chain(state: State) -> State:
    """Generate response using RAG content"""
    if not state["rag_results"]:
        print("Warning: rag_chain called without rag_results. Rerouting to LLM fallback.")
        state["should_reroute_to_llm"] = True
        return state
    
    try:
        context = "\n\n".join([doc.page_content for doc in state["rag_results"]])
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Use the following **retrieved context** to answer the user's question.
        If the retrieved context doesn't contain enough information to answer the question, clearly state that 
        you cannot answer based on the provided information, but do not make up an answer.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query})
        
        if "cannot answer based on the provided information" in response.content.lower():
            print("RAG LLM indicated insufficient context. Setting should_reroute_to_llm to True.")
            state["should_reroute_to_llm"] = True
            # DO NOT append a message here; the LLM fallback will handle it.
        else:
            state["messages"].append(AIMessage(content=response.content))
            state["should_reroute_to_llm"] = False
            
    except Exception as e:
        print(f"Error generating RAG response in rag_chain: {e}")
        state["should_reroute_to_llm"] = True # Fallback on error
    
    return state

def llm_fallback_chain(state: State) -> State:
    """Generate response using OpenAI LLM for general knowledge"""
    try:
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Answer the user's question to the best of your ability 
        using your general knowledge. **Do not mention any lack of context or refer to database information.**
        
        Question: {query}
        
        Answer:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"query": query})
        
        state["messages"].append(AIMessage(content=response.content))
        
    except Exception as e:
        print(f"Error generating LLM fallback response in llm_fallback_chain: {e}")
        state["messages"].append(AIMessage(content="I apologize, but I encountered an error while trying to answer with general knowledge."))
    
    return state

def route_query(state: State) -> str:
    """Route query to RAG or LLM fallback based on content availability"""
    if state["rag_results"]:
        print("Routing to RAG chain.")
        return "rag_chain"
    else:
        print("Routing to LLM fallback chain immediately (no RAG docs).")
        return "llm_fallback_chain"

def check_rag_outcome(state: State) -> str:
    """Check if RAG chain succeeded or needs LLM fallback"""
    if state["should_reroute_to_llm"]:
        print("RAG chain indicated reroute to LLM fallback.")
        return "llm_fallback_chain"
    else:
        print("RAG chain successfully generated a response. Ending process.")
        return "__end__"

# Rebuild the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("scrape_website", scrape_website)
graph_builder.add_node("search_rag_content", search_rag_content)
graph_builder.add_node("process_document", process_document) # Add process_document node
graph_builder.add_node("rag_chain", rag_chain)
graph_builder.add_node("llm_fallback_chain", llm_fallback_chain)
graph_builder.add_node("check_rag_outcome", check_rag_outcome)

# Add edges and conditional logic
graph_builder.add_edge(START, "search_rag_content")

# Conditional edge from search_rag_content to route_query
graph_builder.add_conditional_edges(
    "search_rag_content",
    route_query,
    {
        "rag_chain": "rag_chain",
        "llm_fallback_chain": "llm_fallback_chain",
    },
)

# After rag_chain, check its outcome
graph_builder.add_conditional_edges(
    "rag_chain",
    check_rag_outcome,
    {
        "llm_fallback_chain": "llm_fallback_chain",
        "__end__": END, # If RAG was successful, end the graph
    },
)

graph_builder.add_edge("llm_fallback_chain", END)

# Compile the graph
graph = graph_builder.compile()

# Utility functions for external use
def add_website_content(url: str) -> bool:
    """Add website content to the RAG database"""
    try:
        state = State(
            messages=[],
            query="",
            website_url=url,
            document_path=None,
            rag_results=[],
            should_reroute_to_llm=False
        )
        
        # Debug log for website URL being added
        print(f"DEBUG: add_website_content called for URL: {url}")

        canonical_url = _canonicalize_url(url)
        state["website_url"] = canonical_url # Update state with canonical URL
        print(f"DEBUG: add_website_content processed canonical URL: {canonical_url}")
        
        # Scrape and store content
        state = scrape_website(state)
        return True
        
    except Exception as e:
        print(f"Error adding website content: {e}")
        return False

def add_document_content(file_path: str) -> bool:
    """Add document content to the RAG database"""
    try:
        state = State(
            messages=[],
            query="",
            website_url=None,
            document_path=file_path,
            rag_results=[],
            should_reroute_to_llm=False
        )
        
        # Debug log for document path being added
        print(f"DEBUG: add_document_content called for file_path: {file_path}")

        canonical_file_path = _normalize_document_path(file_path)
        state["document_path"] = canonical_file_path # Update state with normalized path
        print(f"DEBUG: add_document_content processed normalized file_path: {canonical_file_path}")
        
        # Process and store document
        state = process_document(state)
        return True
        
    except Exception as e:
        print(f"Error adding document content: {e}")
        return False

def query_rag(query: str, website_url: str = None) -> str:
    """Query the RAG system"""
    try:
        # Ensure messages are initialized correctly for graph invocation
        initial_messages = [HumanMessage(content=query)]
        state = State(
            messages=initial_messages,
            query=query,
            website_url=website_url,
            document_path=None, # Initialize document_path
            rag_results=[], 
            should_reroute_to_llm=False 
        )

        # If website_url is provided, scrape it first
        if website_url:
            state = scrape_website(state)

        # Invoke the graph
        final_state = graph.invoke(state)

        # Return the content of the last AI message
        if final_state["messages"]:
            return final_state["messages"][-1].content
        else:
            return "Sorry, I couldn't generate a response."

    except Exception as e:
        print(f"Error querying RAG system: {e}")
        return f"An internal error occurred: {str(e)}. Please try again."

def get_available_content_sources() -> List[str]:
    """Get list of websites and documents in the database"""
    try:
        # Directly query the ChromaDB collection to see raw stored data
        collection = vectorstore._collection
        raw_docs_in_db = collection.get(include=["metadatas"])
        
        print(f"DEBUG: ChromaDB Collection contains {len(raw_docs_in_db['ids'])} items.")
        for i, doc_id in enumerate(raw_docs_in_db['ids']):
            metadata = raw_docs_in_db['metadatas'][i]
            print(f"DEBUG: DB Item ID: {doc_id}, Metadata: {metadata}")

        sources = set()
        if raw_docs_in_db and 'metadatas' in raw_docs_in_db:
            for metadata in raw_docs_in_db['metadatas']:
                if metadata and 'source' in metadata:
                    source = metadata['source']
                    
                    # Canonicalize URLs for display, but keep original for documents
                    if source.startswith(("http://", "https://")):
                        display_source = _canonicalize_url(source)
                    else:
                        # For documents, ensure consistent display and match delete logic
                        display_source = _normalize_document_path(source)
                    
                    sources.add(display_source) # Add the canonicalized/normalized source for display
                    print(f"DEBUG: Retrieved content source (original: {source}, display: {display_source})") # Debug log
        
        return list(sources) if sources else ["No content stored yet"]
    except Exception as e:
        print(f"Error getting available content sources: {e}")
        return ["Error retrieving content sources"]

def delete_content_source(source_to_delete: str) -> bool:
    """Delete all content associated with a given source from the RAG database"""
    try:
        print(f"Attempting to delete content for source: {source_to_delete}")

        # Add a debug print of the current collection size before deletion attempt
        collection_before_delete = vectorstore._collection.count()
        print(f"DEBUG: ChromaDB collection size before delete attempt: {collection_before_delete}")

        # Apply canonicalization if it's a URL, or normalization if it's a document path
        if source_to_delete.startswith(("http://", "https://")):
            canonical_source_to_delete = _canonicalize_url(source_to_delete)
            print(f"DEBUG: Canonicalized source for deletion (URL): {canonical_source_to_delete}")
        else:
            # Assume it's a document path and normalize it
            canonical_source_to_delete = _normalize_document_path(source_to_delete)
            print(f"DEBUG: Normalized source for deletion (Document): {canonical_source_to_delete}")

        # ChromaDB's delete method can filter by metadata directly
        vectorstore._collection.delete(where={"source": canonical_source_to_delete})
        
        # Check collection size after deletion to confirm if items were removed
        collection_after_delete = vectorstore._collection.count()
        print(f"DEBUG: ChromaDB collection size after delete attempt: {collection_after_delete}")

        if collection_after_delete < collection_before_delete:
            print(f"Successfully deleted items for source: {canonical_source_to_delete}") # Debug log
            return True
        else:
            print(f"No items were deleted (or collection size unchanged) for source: {canonical_source_to_delete}") # Debug log
            return False
    except Exception as e:
        print(f"Error deleting content for source {source_to_delete}: {e}")
        return False