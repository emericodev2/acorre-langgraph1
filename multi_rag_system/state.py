# multi_rag_system/state.py
"""State definitions for the Multi-RAG system."""

from typing import List, Optional, TypedDict, Annotated
from langchain.schema import Document
import operator

class AgentState(TypedDict):
    """State of the Multi-RAG agent."""
    # Core state
    query: str
    messages: Annotated[list, operator.add]
    
    # RAG results
    website_results: Optional[List[Document]]
    document_results: Optional[List[Document]]
    llm_search_results: Optional[str]
    
    # Processing metadata
    final_answer: Optional[str]
    current_step: str
    confidence_score: float
    source_metadata: dict
    
    # Configuration
    max_tokens: int
    temperature: float