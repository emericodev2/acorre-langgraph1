import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

load_dotenv()

def main():
    """Main entry point for the RAG system"""
    print("🤖 Website RAG Agent System")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or export it:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        from examples.usage_examples import (
            demonstrate_full_workflow,
            demonstrate_separate_agents,
            demonstrate_batch_processing
        )
        
        # Run demonstrations
        demonstrate_full_workflow()
        demonstrate_separate_agents()
        demonstrate_batch_processing()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have all the required packages installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()