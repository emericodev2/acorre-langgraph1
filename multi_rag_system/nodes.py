# multi_rag_system/nodes.py
"""Node implementations for the Multi-RAG workflow."""

import logging
from .state import AgentState
from .system import MultiRAGSystem
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Global system instance - will be initialized when needed
_rag_system = None

def get_rag_system() -> MultiRAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = MultiRAGSystem()
    return _rag_system

async def website_rag_node(state: AgentState) -> AgentState:
    """Search website RAG."""
    logger.info("Searching website RAG...")
    rag_system = get_rag_system()
    
    results, confidence = await rag_system.search_websites(state["query"])
    
    state["website_results"] = results
    state["current_step"] = "website_rag_complete"
    state["confidence_score"] = confidence
    state["source_metadata"]["website_chunks"] = len(results)
    
    logger.info(f"Found {len(results)} relevant website documents, confidence: {confidence:.2f}")
    return state

async def document_rag_node(state: AgentState) -> AgentState:
    """Search document RAG."""
    logger.info("Searching document RAG...")
    rag_system = get_rag_system()
    
    results, confidence = await rag_system.search_documents(state["query"])
    
    state["document_results"] = results
    state["current_step"] = "document_rag_complete"
    
    # Update confidence with the maximum of existing and new confidence
    state["confidence_score"] = max(state.get("confidence_score", 0.0), confidence)
    state["source_metadata"]["document_chunks"] = len(results)
    
    logger.info(f"Found {len(results)} relevant documents, confidence: {confidence:.2f}")
    return state

async def llm_search_node(state: AgentState) -> AgentState:
    """Perform LLM search as fallback."""
    logger.info("Performing LLM search...")
    rag_system = get_rag_system()
    
    result, confidence = await rag_system.llm_search(state["query"])
    
    state["llm_search_results"] = result
    state["current_step"] = "llm_search_complete"
    state["confidence_score"] = max(state.get("confidence_score", 0.0), confidence)
    state["source_metadata"]["llm_search_used"] = True
    
    logger.info("LLM search completed")
    return state

async def synthesize_answer_node(state: AgentState) -> AgentState:
    """Synthesize final answer from all sources."""
    logger.info("Synthesizing final answer...")
    rag_system = get_rag_system()
    
    website_results = state.get("website_results", [])
    document_results = state.get("document_results", [])
    llm_results = state.get("llm_search_results")
    
    final_answer = await rag_system.synthesize_answer(
        state["query"],
        website_results,
        document_results,
        llm_results
    )
    
    state["final_answer"] = final_answer
    state["current_step"] = "synthesis_complete"
    
    # Add the final answer to messages
    state["messages"].append(AIMessage(content=final_answer))
    
    # Update source metadata
    state["source_metadata"].update({
        "total_sources": len(website_results) + len(document_results) + (1 if llm_results else 0),
        "final_confidence": state["confidence_score"]
    })
    
    logger.info("Answer synthesis completed")
    return state
