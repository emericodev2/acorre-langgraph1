# multi_rag_system/routing.py
"""Routing logic for the Multi-RAG workflow."""

from .state import AgentState

def should_try_document_rag(state: AgentState) -> str:
    """Decide whether to try document RAG."""
    website_results = state.get("website_results", [])
    confidence = state.get("confidence_score", 0.0)
    
    # Try document RAG if website results are insufficient
    if len(website_results) < 2 or confidence < 0.8:
        return "try_document"
    return "sufficient"

def should_try_llm_search(state: AgentState) -> str:
    """Decide whether to try LLM search."""
    website_results = state.get("website_results", [])
    document_results = state.get("document_results", [])
    confidence = state.get("confidence_score", 0.0)
    
    # Try LLM search if both RAG sources are insufficient
    total_results = len(website_results) + len(document_results)
    if total_results < 2 or confidence < 0.7:
        return "try_llm"
    return "sufficient"
