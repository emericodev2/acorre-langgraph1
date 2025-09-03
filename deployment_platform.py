#deployment_platform.py - Platform-specific deployment utilities
import os
import json
import asyncio
from typing import Dict, Any, List
from multi_rag_system.graph import run_query
from multi_rag_system.system import MultiRAGSystem

class PlatformDeployment:
    """Utilities for LangGraph Platform deployment."""
    
    def __init__(self):
        self.rag_system = MultiRAGSystem()
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize the RAG system with default content."""
        
        # Example websites to ingest on startup
        default_websites = os.getenv("DEFAULT_WEBSITES", "https://emerico.com").split(",")
        default_websites = [url.strip() for url in default_websites if url.strip()]
        
        results = {"websites": None, "documents": None}
        
        if default_websites:
            try:
                results["websites"] = await self.rag_system.ingest_websites(default_websites)
                print(f"✓ Ingested {results['websites']['successful']} websites on startup")
            except Exception as e:
                print(f"✗ Failed to ingest websites on startup: {e}")
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Platform health check endpoint."""
        try:
            # Test a simple query
            test_result = await run_query("Hello, are you working?")
            
            return {
                "status": "healthy",
                "system_ready": bool(test_result.get("answer")),
                "website_vectorstore": self.rag_system.website_vectorstore is not None,
                "document_vectorstore": self.rag_system.document_vectorstore is not None,
                "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
                "test_query_success": bool(test_result.get("answer"))
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "system_ready": False
            }
    
    async def ingest_content(self, content_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest content based on configuration."""
        results = {}
        
        # Ingest websites if provided
        if "websites" in content_config:
            try:
                results["websites"] = await self.rag_system.ingest_websites(
                    content_config["websites"]
                )
            except Exception as e:
                results["websites"] = {"error": str(e)}
        
        # Ingest documents if provided
        if "documents" in content_config:
            try:
                results["documents"] = await self.rag_system.ingest_documents(
                    content_config["documents"]
                )
            except Exception as e:
                results["documents"] = {"error": str(e)}
        
        return results

# Initialize deployment instance
deployment = PlatformDeployment()

# Platform startup hook
async def startup():
    """Platform startup initialization."""
    print("🚀 Initializing Multi-RAG System on LangGraph Platform...")
    
    try:
        results = await deployment.initialize_system()
        print(f"✅ System initialized: {results}")
        
        # Perform health check
        health = await deployment.health_check()
        print(f"🏥 Health check: {health}")
        
        if health["status"] == "healthy":
            print("🎉 Multi-RAG System ready for queries!")
        else:
            print("⚠️  System may not be fully ready")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")

# Run startup if executed directly
if __name__ == "__main__":
    asyncio.run(startup())