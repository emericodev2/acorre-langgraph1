#!/usr/bin/env python3
"""
Debug script to get detailed error information from LangGraph runs
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:2024"

def create_assistant_and_run_with_debug(graph_name, input_data):
    """Create assistant, run with input, and get detailed error info"""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING {graph_name.upper()} GRAPH")
    print(f"{'='*60}")
    
    # Create assistant
    print(f"📝 Creating assistant for graph: {graph_name}")
    assistant_payload = {
        "graph_id": graph_name,
        "config": {},
        "metadata": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/assistants", json=assistant_payload)
        response.raise_for_status()
        assistant = response.json()
        assistant_id = assistant.get("assistant_id")
        print(f"✅ Assistant created: {assistant_id}")
    except Exception as e:
        print(f"❌ Failed to create assistant: {e}")
        return
    
    # Create thread
    print("🧵 Creating thread...")
    try:
        response = requests.post(f"{BASE_URL}/threads", json={"metadata": {}})
        response.raise_for_status()
        thread = response.json()
        thread_id = thread.get("thread_id")
        print(f"✅ Thread created: {thread_id}")
    except Exception as e:
        print(f"❌ Failed to create thread: {e}")
        return
    
    # Run assistant
    print(f"🚀 Running assistant with input:")
    print(json.dumps(input_data, indent=2))
    
    run_payload = {
        "assistant_id": assistant_id,
        "input": input_data,
        "config": {},
        "metadata": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/threads/{thread_id}/runs", json=run_payload)
        response.raise_for_status()
        run = response.json()
        run_id = run.get("run_id")
        print(f"✅ Run started: {run_id}")
    except Exception as e:
        print(f"❌ Failed to start run: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return
    
    # Monitor run with detailed output
    print(f"⏳ Monitoring run progress...")
    start_time = time.time()
    max_wait = 120  # Increased timeout
    
    while time.time() - start_time < max_wait:
        try:
            # Get run status
            response = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}")
            response.raise_for_status()
            run = response.json()
            
            status = run.get("status")
            print(f"📊 Status: {status}")
            
            # If there's an error, get detailed info
            if status == "error":
                print("🚨 ERROR DETAILS:")
                print(f"Run object: {json.dumps(run, indent=2)}")
                
                # Try to get thread state for more info
                try:
                    state_response = requests.get(f"{BASE_URL}/threads/{thread_id}/state")
                    if state_response.status_code == 200:
                        state = state_response.json()
                        print(f"Thread state: {json.dumps(state, indent=2)}")
                except:
                    pass
                
                # Try to get run history
                try:
                    history_response = requests.get(f"{BASE_URL}/threads/{thread_id}/history")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        print(f"Thread history: {json.dumps(history, indent=2)}")
                except:
                    pass
                
                break
            
            elif status == "success":
                print("✅ Run completed successfully!")
                # Get final state
                try:
                    state_response = requests.get(f"{BASE_URL}/threads/{thread_id}/state")
                    if state_response.status_code == 200:
                        state = state_response.json()
                        print(f"Final state: {json.dumps(state, indent=2)}")
                except:
                    pass
                break
            
            elif status in ["cancelled"]:
                print(f"🛑 Run {status}")
                break
            
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Error checking run: {e}")
            break
    
    if time.time() - start_time >= max_wait:
        print(f"⏰ Run timed out after {max_wait} seconds")

def check_environment():
    """Check if environment variables are properly set"""
    print("🔧 Checking environment setup...")
    
    # Try to get info endpoint
    try:
        response = requests.get(f"{BASE_URL}/info")
        if response.status_code == 200:
            info = response.json()
            print("ℹ️ Server info:")
            print(json.dumps(info, indent=2))
    except Exception as e:
        print(f"❌ Could not get server info: {e}")

def test_simple_ingest():
    """Test ingestion with minimal data"""
    input_data = {
        "url": "https://httpbin.org/json",  # Simple JSON endpoint for testing
        "file_path": None,
        "source_type": "",
        "source_name": "",
        "docs": [],
        "chunks": []
    }
    
    create_assistant_and_run_with_debug("ingest", input_data)

def test_simple_query():
    """Test query with minimal data"""
    input_data = {
        "query": "Hello, what can you tell me?",
        "source_type": None,
        "source_name": None,
        "answer": ""
    }
    
    create_assistant_and_run_with_debug("query", input_data)

def main():
    print("🐛 LangGraph Debug Session")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Test with simpler data first
    print("\n🧪 Testing with simple data...")
    
    # Test ingestion
    test_simple_ingest()
    
    # Test query
    test_simple_query()

if __name__ == "__main__":
    main()