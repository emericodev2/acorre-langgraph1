import os
import re
import uuid
import requests
from typing import List, Dict, Optional, TypedDict, Literal
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

# ==================== CONFIGURATION ====================
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTOR_DB_PATH = "./chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SEARCH_K = 6
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-3.5-turbo"
    REQUEST_TIMEOUT = 30
    USER_AGENT = "Mozilla/5.0 (compatible; RAG-Agent/1.0)"

config = Config()

# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    url: str
    question: str
    collection_name: Optional[str]
    documents: List[Document]
    vector_store: Optional[Chroma]
    answer: str
    sources: List[str]
    error: str
    status: Literal["pending", "processing", "success", "error"]
    metadata: Dict

# ==================== UTILITY FUNCTIONS ====================
def generate_collection_name(url: str) -> str:
    """Generate a unique collection name from URL"""
    domain = re.sub(r'https?://(www\.)?', '', url.split('/')[2])
    domain = re.sub(r'[^a-zA-Z0-9]', '_', domain)
    unique_id = uuid.uuid4().hex[:8]
    return f"web_rag_{domain}_{unique_id}"

def clean_html_content(html: str) -> str:
    """Clean HTML content and extract text"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        element.decompose()
    
    # Get text and clean it
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'(([A-Z0-9][A-Z0-9_-]*)(\.[A-Z0-9][A-Z0-9_-]*)+)'  # domain
        r'(:[0-9]+)?'  # port
        r'(/.*)?$', re.IGNORECASE)  # path
    return bool(pattern.match(url))

def get_page_title(url: str) -> str:
    """Extract page title from URL"""
    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT, 
                              headers={'User-Agent': config.USER_AGENT})
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        return title.get_text().strip() if title else "Untitled"
    except:
        return "Unknown Title"

def save_metadata(collection_name: str, url: str, document_count: int):
    """Save collection metadata to file"""
    metadata_path = Path(config.VECTOR_DB_PATH) / "metadata.json"
    metadata_path.parent.mkdir(exist_ok=True, parents=True)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    metadata[collection_name] = {
        "url": url,
        "document_count": document_count,
        "created_at": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        "title": get_page_title(url)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# ==================== AGENT NODES ====================
class RAGAgents:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def validate_inputs(self, state: AgentState) -> Dict:
        """Validate URL and question inputs"""
        errors = []
        
        if not state.get("url"):
            errors.append("URL is required")
        elif not validate_url(state["url"]):
            errors.append("Invalid URL format")
        
        if not state.get("question"):
            errors.append("Question is required")
        
        return {
            "error": "; ".join(errors) if errors else "",
            "status": "error" if errors else "processing",
            "metadata": {**state.get("metadata", {}), "validation_errors": errors}
        }
    
    def scrape_website(self, state: AgentState) -> Dict:
        """Scrape website content"""
        try:
            print(f"🌐 Scraping: {state['url']}")
            loader = WebBaseLoader(
                state["url"],
                requests_kwargs={
                    'timeout': config.REQUEST_TIMEOUT,
                    'headers': {'User-Agent': config.USER_AGENT}
                }
            )
            
            documents = loader.load()
            page_title = get_page_title(state["url"])
            
            # Enhance documents with metadata
            for doc in documents:
                doc.metadata.update({
                    "source_url": state["url"],
                    "title": page_title,
                    "ingestion_timestamp": "2024-01-01T00:00:00Z"
                })
            
            return {
                "documents": documents,
                "error": "",
                "status": "processing",
                "metadata": {**state.get("metadata", {}), "scraped_docs": len(documents)}
            }
            
        except Exception as e:
            return {
                "documents": [],
                "error": f"Scraping failed: {str(e)}",
                "status": "error"
            }
    
    def process_documents(self, state: AgentState) -> Dict:
        """Process and split documents"""
        try:
            if not state.get("documents"):
                return {
                    "documents": [],
                    "error": "No documents to process",
                    "status": "error"
                }
            
            print(f"📄 Processing {len(state['documents'])} documents...")
            
            # Split documents
            chunks = self.text_splitter.split_documents(state["documents"])
            
            # Clean content
            for chunk in chunks:
                chunk.page_content = clean_html_content(chunk.page_content)
            
            return {
                "documents": chunks,
                "error": "",
                "status": "processing",
                "metadata": {**state.get("metadata", {}), "processed_chunks": len(chunks)}
            }
            
        except Exception as e:
            return {
                "documents": [],
                "error": f"Processing failed: {str(e)}",
                "status": "error"
            }
    
    def store_to_vector_db(self, state: AgentState) -> Dict:
        """Store documents in vector database"""
        try:
            if not state.get("documents"):
                return {
                    "vector_store": None,
                    "error": "No documents to store",
                    "status": "error"
                }
            
            collection_name = state.get("collection_name") or generate_collection_name(state["url"])
            
            print(f"💾 Storing {len(state['documents'])} chunks to vector DB...")
            
            vector_store = Chroma.from_documents(
                documents=state["documents"],
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=config.VECTOR_DB_PATH
            )
            
            # Save metadata
            save_metadata(collection_name, state["url"], len(state["documents"]))
            
            return {
                "vector_store": vector_store,
                "collection_name": collection_name,
                "error": "",
                "status": "processing",
                "metadata": {**state.get("metadata", {}), "collection_name": collection_name}
            }
            
        except Exception as e:
            return {
                "vector_store": None,
                "error": f"Storage failed: {str(e)}",
                "status": "error"
            }
    
    def answer_question(self, state: AgentState) -> Dict:
        """Answer question using RAG"""
        try:
            if not state.get("vector_store"):
                return {
                    "answer": "",
                    "sources": [],
                    "error": "No vector store available",
                    "status": "error"
                }
            
            print(f"🤔 Answering: {state['question']}")
            
            # Create retriever
            retriever = state["vector_store"].as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.SEARCH_K}
            )
            
            # Retrieve relevant documents
            relevant_docs = retriever.invoke(state["question"])
            
            # Format context
            def format_docs(docs):
                formatted = []
                for doc in docs:
                    source = doc.metadata.get("source_url", "Unknown source")
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    formatted.append(f"[Source: {source}]\n{content}")
                return "\n\n---\n\n".join(formatted)
            
            # Create QA chain
            template = """You are an assistant for question-answering tasks. 
            Use the following retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know.
            Be concise and accurate.

            Context: {context}

            Question: {question}

            Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = qa_chain.invoke(state["question"])
            
            # Extract sources
            sources = list(set(doc.metadata.get("source_url", "Unknown") for doc in relevant_docs))
            
            return {
                "answer": answer,
                "sources": sources,
                "error": "",
                "status": "success",
                "metadata": {**state.get("metadata", {}), "relevant_docs": len(relevant_docs)}
            }
            
        except Exception as e:
            return {
                "answer": "",
                "sources": [],
                "error": f"QA failed: {str(e)}",
                "status": "error"
            }
    
    def handle_error(self, state: AgentState) -> Dict:
        """Handle errors gracefully"""
        error_msg = state.get("error", "Unknown error occurred")
        print(f"❌ Error: {error_msg}")
        
        return {
            "error": error_msg,
            "status": "error",
            "answer": f"I encountered an error: {error_msg}",
            "metadata": {**state.get("metadata", {}), "error_occurred": True}
        }

# ==================== GRAPH WORKFLOW ====================
class WebsiteRAGGraph:
    def __init__(self):
        self.agents = RAGAgents()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("validate_inputs", self.agents.validate_inputs)
        workflow.add_node("scrape_website", self.agents.scrape_website)
        workflow.add_node("process_documents", self.agents.process_documents)
        workflow.add_node("store_to_vector_db", self.agents.store_to_vector_db)
        workflow.add_node("answer_question", self.agents.answer_question)
        workflow.add_node("handle_error", self.agents.handle_error)
        
        # Define edges
        workflow.set_entry_point("validate_inputs")
        
        # Conditional edges from validation
        workflow.add_conditional_edges(
            "validate_inputs",
            self._check_validation,
            {
                "valid": "scrape_website",
                "invalid": "handle_error"
            }
        )
        
        # Linear flow for successful processing
        workflow.add_edge("scrape_website", "process_documents")
        workflow.add_edge("process_documents", "store_to_vector_db")
        workflow.add_edge("store_to_vector_db", "answer_question")
        
        # Error handling
        workflow.add_conditional_edges(
            "scrape_website",
            self._check_success,
            {"continue": "process_documents", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "process_documents",
            self._check_success,
            {"continue": "store_to_vector_db", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "store_to_vector_db",
            self._check_success,
            {"continue": "answer_question", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "answer_question",
            self._check_success,
            {"success": END, "error": "handle_error"}
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _check_validation(self, state: AgentState) -> Literal["valid", "invalid"]:
        """Check if validation passed"""
        return "valid" if not state.get("error") else "invalid"
    
    def _check_success(self, state: AgentState) -> Literal["continue", "error", "success"]:
        """Check if operation was successful"""
        if state.get("status") == "error":
            return "error"
        elif state.get("status") == "success":
            return "success"
        return "continue"
    
    def run_workflow(self, url: str, question: str, collection_name: Optional[str] = None) -> Dict:
        """Run the complete RAG workflow"""
        initial_state = AgentState(
            url=url,
            question=question,
            collection_name=collection_name,
            documents=[],
            vector_store=None,
            answer="",
            sources=[],
            error="",
            status="pending",
            metadata={"workflow_started": True}
        )
        
        print("🚀 Starting RAG Workflow...")
        result = self.graph.invoke(initial_state)
        
        # Clean up the result for better readability
        cleaned_result = {
            "success": result["status"] == "success",
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "error": result.get("error", ""),
            "collection_name": result.get("collection_name"),
            "document_count": len(result.get("documents", [])),
            "metadata": result.get("metadata", {})
        }
        
        return cleaned_result

# ==================== MAIN EXECUTION ====================
def main():
    """Main function to demonstrate the RAG system"""
    print("🤖 Website RAG Agent with LangGraph")
    print("=" * 40)
    
    if not config.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Initialize the graph system
    rag_system = WebsiteRAGGraph()
    
    # Example 1: Full workflow with Wikipedia
    print("\n" + "="*60)
    print("📚 Example 1: AI Wikipedia RAG")
    print("="*60)
    
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    question = "What are the main applications of artificial intelligence?"
    
    result = rag_system.run_workflow(url, question)
    
    print(f"\n🌐 URL: {url}")
    print(f"❓ Question: {question}")
    print(f"✅ Success: {result['success']}")
    
    if result['success']:
        print(f"📝 Answer: {result['answer'][:200]}...")
        print(f"📚 Sources: {result['sources']}")
        print(f"📊 Documents processed: {result['document_count']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Example 2: Machine Learning
    print("\n" + "="*60)
    print("🤖 Example 2: Machine Learning RAG")
    print("="*60)
    
    url2 = "https://en.wikipedia.org/wiki/Machine_learning"
    question2 = "What are the different types of machine learning?"
    
    result2 = rag_system.run_workflow(url2, question2)
    
    print(f"\n🌐 URL: {url2}")
    print(f"❓ Question: {question2}")
    print(f"✅ Success: {result2['success']}")
    
    if result2['success']:
        print(f"📝 Answer: {result2['answer'][:200]}...")
        print(f"📚 Sources: {result2['sources']}")
    else:
        print(f"❌ Error: {result2['error']}")
    
    # Example 3: Error case (invalid URL)
    print("\n" + "="*60)
    print("⚠️  Example 3: Error Handling")
    print("="*60)
    
    result3 = rag_system.run_workflow("invalid-url", "Test question")
    
    print(f"✅ Success: {result3['success']}")
    if not result3['success']:
        print(f"❌ Error: {result3['error']}")

def interactive_mode():
    """Interactive mode for testing different URLs and questions"""
    print("\n🎮 Interactive RAG Mode")
    print("=" * 30)
    
    rag_system = WebsiteRAGGraph()
    
    while True:
        print("\nEnter a website URL (or 'quit' to exit):")
        url = input("> ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            break
        
        if not validate_url(url):
            print("❌ Invalid URL format. Please try again.")
            continue
        
        print("Enter your question:")
        question = input("> ").strip()
        
        if not question:
            print("❌ Question cannot be empty.")
            continue
        
        print("⏳ Processing...")
        result = rag_system.run_workflow(url, question)
        
        print("\n" + "="*50)
        if result['success']:
            print("✅ SUCCESS!")
            print(f"📝 Answer: {result['answer']}")
            print(f"📚 Sources: {result['sources']}")
            print(f"📊 Documents processed: {result['document_count']}")
        else:
            print("❌ FAILED!")
            print(f"Error: {result['error']}")
        print("="*50)

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not config.OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY in the .env file")
        api_key = input("Or enter it now: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            config.OPENAI_API_KEY = api_key
        else:
            print("API key is required. Exiting.")
            exit(1)
    
    # Run demonstrations
    main()
    
    # Uncomment the next line to enable interactive mode
    interactive_mode()