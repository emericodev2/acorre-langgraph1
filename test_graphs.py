#!/usr/bin/env python3
"""
Simplified LangGraph test script
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:2024"

def run_graph(graph_name, input_data):
    """Run a graph with input data and return result"""
    print(f"\n🔄 Running {graph_name} graph...")
    
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
    for _ in range(30):  # 90 seconds max
        run_status = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}").json()
        status = run_status["status"]
        
        print(f"Status: {status}")
        
        if status == "success":
            # Get result
            state = requests.get(f"{BASE_URL}/threads/{thread_id}/state").json()
            return state["values"]
        
        elif status == "error":
            print(f"❌ Error: {run_status.get('error', 'Unknown error')}")
            return None
            
        time.sleep(3)
    
    print("❌ Timeout")
    return None

def main():
    print("🚀 Simple LangGraph Test")
    
    # Test query only (assumes data already ingested)
    query_input = {
        "query": "What is emerico?",
        "source_type": None,
        "source_name": None,
        "answer": ""
    }
    
    result = run_graph("query", query_input)
    if result:
        answer = result.get('answer', 'No answer')
        print(f"✅ Complete Answer:\n{answer}")

if __name__ == "__main__":
    main()