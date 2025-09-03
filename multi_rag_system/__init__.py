# multi_rag_system/__init__.py
"""Multi-RAG System for LangGraph Platform deployment."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .graph import graph
from .system import MultiRAGSystem
from .state import AgentState

__all__ = ["graph", "MultiRAGSystem", "AgentState"]