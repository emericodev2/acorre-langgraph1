import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph.workflow import WebsiteRAGWorkflow
from agents.ingestion_agent import DataIngestionAgent
from agents.qa_agent import QAAgent

def run_demo():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    question = "What are the main applications of artificial intelligence?"
    
    workflow = WebsiteRAGWorkflow()
    result = workflow.run_workflow(url, question)
    
    if result["success"]:
        print("✅ Answer:", result["answer"])
        print("📚 Sources:", result["sources"])
    else:
        print("❌ Error:", result["error"])

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Enter OpenAI API key: ")
    run_demo()
