# multi_rag_system/graph.py
"""LangGraph workflow definition for Multi-RAG system."""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .state import AgentState
from .nodes import (
    website_rag_node,
    document_rag_node,
    llm_search_node,
    synthesize_answer_node
)
from .routing import should_try_document_rag, should_try_llm_search

def create_graph() -> StateGraph:
    """Create the Multi-RAG workflow graph."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("website_rag", website_rag_node)
    workflow.add_node("document_rag", document_rag_node)
    workflow.add_node("llm_search", llm_search_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    
    # Set entry point
    workflow.set_entry_point("website_rag")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "website_rag",
        should_try_document_rag,
        {
            "try_document": "document_rag",
            "sufficient": "synthesize_answer"
        }
    )
    
    workflow.add_conditional_edges(
        "document_rag",
        should_try_llm_search,
        {
            "try_llm": "llm_search",
            "sufficient": "synthesize_answer"
        }
    )
    
    workflow.add_edge("llm_search", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)
    
    return workflow.compile()

# Create the graph instance for LangGraph Platform
graph = create_graph()

# Platform-compatible invoke function
async def run_query(query: str, **kwargs) -> dict:
    """Run a query through the Multi-RAG system."""
    
    initial_state = AgentState(
        query=query,
        messages=[HumanMessage(content=query)],
        website_results=None,
        document_results=None,
        llm_search_results=None,
        final_answer=None,
        current_step="starting",
        confidence_score=0.0,
        source_metadata={
            "website_chunks": 0,
            "document_chunks": 0,
            "llm_search_used": False
        },
        max_tokens=kwargs.get("max_tokens", 4000),
        temperature=kwargs.get("temperature", 0.1)
    )
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    return {
        "query": query,
        "answer": final_state.get("final_answer"),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "sources_used": final_state.get("source_metadata", {}),
        "processing_steps": final_state.get("current_step"),
        "messages": final_state.get("messages", [])
    }