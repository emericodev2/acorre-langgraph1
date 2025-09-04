#!/usr/bin/env python3
"""
Sample RAG operations for LangGraph
"""

import requests
import time

BASE_URL = "http://127.0.0.1:2024"

def run_graph(graph_name, input_data, description=""):
    """Run a graph with input data and return result"""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    
    try:
        # Create assistant
        assistant = requests.post(f"{BASE_URL}/assistants", json={
            "graph_id": graph_name,
            "config": {},
            "metadata": {}
        }).json()
        assistant_id = assistant["assistant_id"]
        
        # Create thread
        thread = requests.post(f"{BASE_URL}/threads", json={"metadata": {}}).json()
        thread_id = thread["thread_id"]
        
        # Start run
        run = requests.post(f"{BASE_URL}/threads/{thread_id}/runs", json={
            "assistant_id": assistant_id,
            "input": input_data,
            "config": {},
            "metadata": {}
        }).json()
        run_id = run["run_id"]
        
        # Wait for completion
        print("⏳ Processing...")
        for i in range(40):  # 2 minutes max
            run_status = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}").json()
            status = run_status["status"]
            
            if i % 5 == 0:  # Show status every 15 seconds
                print(f"Status: {status}")
            
            if status == "success":
                # Get result
                state = requests.get(f"{BASE_URL}/threads/{thread_id}/state").json()
                return state["values"]
            
            elif status == "error":
                print(f"❌ Error: {run_status.get('error', 'Unknown error')}")
                return None
                
            time.sleep(3)
        
        print("❌ Timeout after 2 minutes")
        return None
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

# Sample data ingestion functions
def ingest_website(url, description=""):
    """Ingest a website"""
    input_data = {
        "url": url,
        "file_path": None,
        "source_type": "",
        "source_name": "",
        "docs": [],
        "chunks": []
    }
    
    desc = f"Ingesting Website: {url}" + (f" - {description}" if description else "")
    result = run_graph("ingest", input_data, desc)
    
    if result:
        chunks = len(result.get('chunks', []))
        docs = len(result.get('docs', []))
        print(f"✅ Success! Processed {docs} documents into {chunks} chunks")
        print(f"📝 Source: {result.get('source_name', 'Unknown')}")
        return True
    return False

def ingest_file(file_path, description=""):
    """Ingest a local file"""
    input_data = {
        "url": None,
        "file_path": file_path,
        "source_type": "",
        "source_name": "",
        "docs": [],
        "chunks": []
    }
    
    desc = f"Ingesting File: {file_path}" + (f" - {description}" if description else "")
    result = run_graph("ingest", input_data, desc)
    
    if result:
        chunks = len(result.get('chunks', []))
        docs = len(result.get('docs', []))
        print(f"✅ Success! Processed {docs} documents into {chunks} chunks")
        print(f"📝 Source: {result.get('source_name', 'Unknown')}")
        return True
    return False

def query_knowledge_base(question, source_type=None, source_name=None):
    """Query the knowledge base"""
    input_data = {
        "query": question,
        "source_type": source_type,
        "source_name": source_name,
        "answer": ""
    }
    
    filters = ""
    if source_type or source_name:
        filters = f" (filtered by: type={source_type}, name={source_name})"
    
    desc = f"Querying: '{question}'{filters}"
    result = run_graph("query", input_data, desc)
    
    if result:
        answer = result.get('answer', 'No answer provided')
        print(f"✅ Answer:\n{answer}")
        return answer
    return None

def main():
    print("🤖 RAG System Sample Operations")
    print("=" * 60)
    
    # Sample 1: Ingest different types of content
    print("\n📥 SAMPLE INGESTION OPERATIONS")
    
    # Company website
    ingest_website("https://emerico.com", "Company website")
    
    # Documentation site
    ingest_website("https://docs.python.org/3/tutorial/", "Python tutorial")
    
    # News article
    ingest_website("https://httpbin.org/html", "Sample HTML content")
    
    # Wait for indexing
    print("\n⏳ Waiting 15 seconds for indexing to complete...")
    time.sleep(15)
    
    # Sample 2: Different types of queries
    print("\n🔍 SAMPLE QUERY OPERATIONS")
    
    # General query
    query_knowledge_base("What information do you have available?")
    
    # Specific query about company
    query_knowledge_base("What does Emerico do?")
    
    # Technical query
    query_knowledge_base("What is Python?")
    
    # Filtered query - only from website sources
    query_knowledge_base(
        "What services are offered?", 
        source_type="website", 
        source_name="https://emerico.com"
    )
    
    # Sample 3: Follow-up questions
    print("\n🔄 FOLLOW-UP QUERIES")
    
    query_knowledge_base("How can I contact them?")
    query_knowledge_base("What are the main features?")
    query_knowledge_base("Give me a summary of all the content")

if __name__ == "__main__":
    main()